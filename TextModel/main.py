import pandas as pd
from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import *
from dataset import *
from train import *
from args import *
from pickle import Unpickler
from lion_pytorch import Lion
import os


if __name__ == '__main__':
    args = get_args()
    setSeeds()
    
    # user별 train-valid split이 오래 걸려서 pickle 사용함.
    # train, valid pickle data 만들려면 make_dataset.py 실행하면 됨.
    with open("dataset/" + args.dataset_path, "rb") as pkl:
        total = os.path.getsize("dataset/" + args.dataset_path)
        with TQDMBytesReader(pkl, total=total) as pbpkl:
            dataset = Unpickler(pbpkl).load()
        print(f"{args.dataset_path} load done")
    
    language_model = load_language_model(args, baseline=args.baseline)
    train_data, valid_data = dataset.train, dataset.test # TODO: 이 부분 주의
    
    train_dataset = UserTextDataset(args, train_data)
    valid_dataset = UserTextDataset(args, valid_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    if args.with_user:
        print('UserTextModel')
        model = UserTextModel(
            args,
            dataset.max_seq_length,
            dataset.num_user_features,
            language_model
            )
    else:
        print('TextModel')
        model = TextModel(
            args,
            dataset.max_seq_length,
            dataset.num_user_features,
            language_model
            )
    

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    train(args, train_dataloader, valid_dataloader, model, optimizer)