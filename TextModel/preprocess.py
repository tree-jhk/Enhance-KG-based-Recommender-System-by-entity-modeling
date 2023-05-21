import pandas as pd
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
tqdm.pandas()


change_ml_title = [
"Alien 3",
"blood & wine",
"american dream",
"dumb & dumber",
"kama sutra: a tale of love",
"kicking and screaming",
"oscar & lucinda",
"richie rich",
"seven",
"willy wonka and the chocolate factory",
"up close and personal"
]

change_to = [
"Alien³",
"blood and wine",
"american dreamz",
"dumb and dumbe",
"kama sutra - a tale of love",
"kicking & screaming",
"oscar and lucinda",
"ri¢hie ri¢h",
"se7en",
"up close & personal",
"willy wonka & the chocolate factory"
]

change_title_dic = {k:v for k, v in zip(change_ml_title, change_to)}

class DataPreprocess(object):
    def __init__(self, args):
        # self.data is consisted with movies whose movie_title and release_date are same.
        self.data = merge().groupby('user_id').filter(lambda x: len(x) >= 10)
        self.movie_info = self.data[['movie_id', 'overview']].drop_duplicates()
        self.movie_info['len_splited_overview'] = self.movie_info.overview.apply(lambda x:len(x.split()))
        self.sentence_lengths = list(self.movie_info['len_splited_overview'])
        self.margin_length = 10
        self.max_seq_length = int(np.mean(self.sentence_lengths) + 2 * np.std(self.sentence_lengths)) + self.margin_length
        self.tokenizer = load_tokenizer(args, baseline=args.baseline)
        
        self.data['gender'] = self.data['gender'].map({'M':1, 'F':2})
        
        self.user_id_list, self.user_age_list, self.user_gender_list = list(self.data.user_id.unique()), list(self.data.age.unique()), list(self.data.gender.unique())
        
        self.max_user_id = max(self.user_id_list)
        self.max_user_age = max(self.user_age_list)
        self.max_user_gender = max(self.user_gender_list)
        
        # age가 0이거나 gender가 0인 경우 없음.
        self.user_id_list = [value for value in self.user_id_list]
        self.user_age_list = [value + self.max_user_id for value in self.user_age_list]
        self.user_gender_list = [value + self.max_user_id + self.max_user_age for value in self.user_gender_list]
        
        self.user_one_hot_list = self.user_id_list + self.user_age_list + self.user_gender_list
        
        self.num_data = len(self.data)
        self.num_user = len(self.user_id_list)
        self.num_age = len(self.user_age_list)
        self.num_gender = len(self.user_gender_list)
        self.num_user_features = self.num_user + self.num_age + self.num_gender
        
        # one-hot 처리를 위함
        self.userid_2_index = {v: i for i, v in enumerate(self.user_id_list)}
        self.userage_2_index = {v: i for i, v in enumerate(self.user_age_list)}
        self.usergender_2_index = {v: i for i, v in enumerate(self.user_gender_list)}
        
        # overview tokenize
        self.movie_info.loc[:, 'tokenized_overview'] = self.movie_info.overview.apply(lambda x:tokenize(args, x, self.tokenizer, self.max_seq_length))
        # merge tokenized overview with original data
        self.data = pd.merge(self.data, self.movie_info[['movie_id', 'tokenized_overview']], how='inner', on='movie_id')
        self.tokenized_overview = self.data['tokenized_overview'] # (self.num_data, self.max_seq_length) shape의 텐서
        
        # one-hot user's categorical features
        self.one_hot()
        
        # train test split
        test_ratio = args.test_ratio
        self.train, self.test = self.split_by_user(test_ratio=test_ratio)
        if test_ratio == 0.0:
            self.predict= self.to_tensor(self.train)
            self.original_data = self.train
        else:
            # tuple(one_hot_data:torch.tensor, tokenized_text_data:torch.tensor) 형태의 튜플로 변함
            self.train, self.test = self.to_tensor(self.train), self.to_tensor(self.test)

    def one_hot(self):
        # Create empty one-hot arrays for user_id, user_age, and user_gender
        user_id_onehot = np.zeros((self.num_data, self.num_user), dtype=int)
        user_age_onehot = np.zeros((self.num_data, self.num_age), dtype=int)
        user_gender_onehot = np.zeros((self.num_data, self.num_gender), dtype=int)

        # Loop over each row in self.data and set the corresponding one-hot values to 1
        for i, row in self.data.iterrows():
            try:
                user_id_onehot[i, self.userid_2_index[row['user_id']]] = 1
                user_age_onehot[i, self.userage_2_index[row['age'] + self.max_user_id]] = 1
                user_gender_onehot[i, self.usergender_2_index[row['gender'] + self.max_user_id + self.max_user_age]] = 1
            except:
                breakpoint()

        # Add one-hot arrays to self.data DataFrame
        self.data = pd.concat([self.data, pd.DataFrame(user_id_onehot, columns=self.user_id_list)], axis=1)
        self.data = pd.concat([self.data, pd.DataFrame(user_age_onehot, columns=self.user_age_list)], axis=1)
        self.data = pd.concat([self.data, pd.DataFrame(user_gender_onehot, columns=self.user_gender_list)], axis=1)
    
    def split_by_user(self, test_ratio:float=0.1):
        # user_id별로 데이터를 분리하여 train, test dataset으로 나누기
        train = pd.DataFrame()
        test = pd.DataFrame()
        if test_ratio == 0.0:
            for user_id in tqdm(self.data['user_id'].unique()):
                user_df = self.data[self.data['user_id'] == user_id]
                train = pd.concat([train, user_df], ignore_index=True)
            return train, None
        else:
            for user_id in tqdm(self.data['user_id'].unique()):
                user_df = self.data[self.data['user_id'] == user_id]
                train_df, test_df = train_test_split(user_df, test_size=test_ratio, random_state=42)
                train = pd.concat([train, train_df], ignore_index=True)
                test = pd.concat([test, test_df], ignore_index=True)
            return train, test
    
    def to_tensor(self, data:pd.DataFrame):
        one_hot_data = torch.Tensor(data[self.user_one_hot_list].to_numpy())
        tokenized_text_data = torch.stack([torch.tensor(tokenized) for tokenized in data['tokenized_overview'].tolist()])
        rating = torch.Tensor(data['rating'].to_numpy())
        assert len(one_hot_data) == len(tokenized_text_data)
        return (one_hot_data, tokenized_text_data, rating)


def merge(
    ml_movie:pd.DataFrame=None, 
    ml_rating:pd.DataFrame=None, 
    ml_user:pd.DataFrame=None,
    tmdb_movie:pd.DataFrame=None,
    ):
    """
    기본적인 전처리를 포함해서,
    ml_100k 데이터셋과 tmdb 데이터셋 merge:
        - 같은 title & release year 기준으로 merge
    매칭된 데이터와 ml_rating 데이터셋 merge:
        - ml_100k의 movie_id 기준으로 merge
    """
    if ml_movie == None:
        ml_movie = pd.read_csv('../Data/MovieLens/movie.csv')
    if ml_rating == None:
        ml_rating = pd.read_csv('../Data/MovieLens/ratings.csv')
    if ml_user == None:
        ml_user = pd.read_csv('../Data/MovieLens/user.csv')
    if tmdb_movie == None:
        tmdb_movie = pd.read_csv('../Data/TMDB/tmdb_5000_movies.csv')
    
    ml_movie.movie_title = ml_movie.movie_title.apply(lambda x: x.lower())
    tmdb_movie.title = tmdb_movie.title.apply(lambda x: x.lower())
    tmdb_movie.original_title = tmdb_movie.original_title.apply(lambda x: x.lower())
    
    tmdb_movie.loc[tmdb_movie['title'] =='america is still the place', 'original_title'] = 'america is still the place'
    tmdb_movie.loc[tmdb_movie['release_date'].isna(), 'release_date'] = '2015'

    def format_movie_title(data):
        title, date = data['title'], data['release_date']
        
        cleaned_title = re.sub(r'\([^()]*\)', '', title).strip()
        cleaned_date = re.sub(r'[^0-9-]+', '', date).strip()
        
        year = re.findall(r'\d{4}', cleaned_date)
        if year:
            year = year[0]
        else:
            year = ''
        
        formatted_title = cleaned_title + ' (' + year + ')'
        return formatted_title

    def remove_parenthesis(text):
        cleaned_text = re.sub(r'\([^()]*\)', '', text)
        return cleaned_text.strip()

    ml_movie.movie_title = ml_movie.movie_title.map(lambda x: remove_parenthesis(x))
    
    tmdb_movie.drop(columns='title', inplace=True)
    tmdb_movie.rename(columns={'original_title':'title'}, inplace=True)
    ml_movie.rename(columns={'movie_title':'title'}, inplace=True)
    
    def c(x):
        try:
            return change_title_dic[x]
        except:
            return x

    # title 추가 변경 전처리
    ml_movie.title = ml_movie.title.apply(lambda x:c(x))
    
    # tmdb에서 title 같은데, 출시일자가 달라서 다르고 줄거리는 같은 영화는 중복 제거
    tmdb_movie.drop_duplicates(['title'], keep='first', inplace=True)
    
    # ml_movie와 tmdb_movie의 title 같은 경우에 대해 rating과 함께 dataframe 합치기 
    m = pd.merge(ml_movie, tmdb_movie, how='inner', on='title')
    
    mm = pd.merge(ml_rating, m, how='inner', on='movie_id')
    
    target = pd.merge(mm, ml_user, how='inner', on='user_id')
    
    return target