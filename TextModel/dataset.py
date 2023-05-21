import torch
from torch.utils.data import Dataset, DataLoader

class UserTextDataset(Dataset):
    def __init__(self, args, data:tuple):
        super().__init__()
        one_hot_data, tokenized_text_data, rating = data
        assert len(one_hot_data) == len(tokenized_text_data)
        self.one_hot_data = one_hot_data.to(args.device)
        self.tokenized_text_data = tokenized_text_data.to(args.device)
        self.rating = rating.to(args.device)

    def __len__(self):
        return len(self.one_hot_data)

    def __getitem__(self, index):
        one_hot_batch = self.one_hot_data[index]
        tokenized_text_batch = self.tokenized_text_data[index]
        rating = self.rating[index]
        return one_hot_batch, tokenized_text_batch, rating

def collate_fn(batch):
    one_hot_batch, tokenized_text_batch, rating = zip(*batch)
    return torch.stack(one_hot_batch), torch.stack(tokenized_text_batch), torch.stack(rating)