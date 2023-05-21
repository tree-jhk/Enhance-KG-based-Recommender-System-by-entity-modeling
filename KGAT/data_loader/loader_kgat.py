import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data) # KG 만들기
        self.print_info(logging) # entity 수에 대한 정보 출력

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # kg_data: ['h', 'r', 't']형태의 데이터프레임
        
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations # 역순 relation 만들기
        # 역순까지 합치기
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 2 # 왜 2를 더하는지 모름
        self.n_relations = max(kg_data['r']) + 1 # 역순 고려한 관계 수
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1 # entity는 0~?까지 다 1개씩 있는 것임.
        self.n_users_entities = self.n_users + self.n_entities # 총 entity 수

        # d + self.n_entities: item의 entity까지 고려해서 padding
        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        # np.unique: 중복 제거
        # k + self.n_entities: item의 entity까지 고려해서 padding
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        # len(train) by 3 행렬 만들기
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0] # user
        cf2kg_train_data['t'] = self.cf_train_data[1] # item

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1] # item
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0] # user

        """
        self.kg_train_data:
            kg_data (이것도 역순이 고려됐음)
                h -> r -> t
                item_entity_A -> relation -> item_entity_B
                item_entity_B -> relation -> item_entity_A
            cf2kg_train_data
                h -> r -> t
                user -> interact -> item
            inverse_cf2kg_train_data
                h -> r -> t
                item -> interact -> user
            
        만들어진 행렬: len(kg_data) + len(cf2kg_train_data) + len(inverse_cf2kg_train_data) by 3
        """
        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        # user-item 관계와 inverse 관계 포함한 모든 relation 수
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        """
        self.train_kg_dict:
            h:[(t, r)]
        self.train_relation_dict:
            r:[(h, t)]
        """
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            # 딕셔너리 만들어야 entity와 relation에 모두 접근 가능
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        # LongTensor로 만들기
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        """
        self.train_kg_dict:
            h:[(t, r)]
        self.train_relation_dict:
            r:[(h, t)]
        """
        # r:[(h, t)]
        for r, ht_list in self.train_relation_dict.items():
            # rows: head
            rows = [e[0] for e in ht_list]
            # rows: tail
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            """
            coo_matrix((data, (i, j)), shape=(M, N)])

            아래와 같은 3개의 배열을 이용해 만든다.

            data[:] 는 순서에 상관없이 matrix 전체를 이용
            i[:]는 matrix의 행 색인을 이용
            j[:]는 matrix의 열 색인을 이용
            """
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj
        # 결국, r:[(h, t)]를 인접행렬 형태로 만든 것일 뿐임.
        # r:h와 t를 담은 인접행렬


    def create_laplacian_dict(self):
        # 수식 구현
        # L_sym = D^(-1/2)LD^(-1/2) = I - D^(-1/2)AD^(-1/2)
        """
        D: d_mat_inv_sqrt
        L: adj
        """
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        # 수식 구현
        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)
        # 결국, r:[(h, t)]를 라플라시안행렬 형태로 만든 것일 뿐임.
        # r:h와 t를 담은 라플라시안행렬

        """
        A_in = sum(self.laplacian_dict.values()):
            그래프 전체의 라플라시안 행렬
        특정 관계가 아닌 전체 그래프의 속성을 나타냄.
        그래프 컨볼루션 네트워크(GCN)의 맥락에서 
        전체 그래프의 라플라시안 행렬은 종종 GCN 계층에 대한 입력으로 사용된다.
        """
        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


