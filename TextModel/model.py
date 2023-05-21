import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings(action="ignore")
torch.set_printoptions(sci_mode=True)

class TextModel(nn.Module):
    def __init__(self, args, max_seq_length, n_user_features, text_embedding_model) -> None:
        super(TextModel, self).__init__()
        self.text_embedding = text_embedding_model.to(args.device)
        self.llm_hidden_size = text_embedding_model.config.hidden_size
        self.text_linear = nn.Linear(self.llm_hidden_size, args.emb_size).to(args.device)

        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.elu = torch.nn.ELU()
        
        self.tmp = nn.Linear(args.emb_size,1).to(args.device)
        self.tmp2 = nn.Linear(max_seq_length, 1).to(args.device)
        
    def forward(self, one_hot_data, tokenized_text_data):
        
        tokenized_text_data = tokenized_text_data.squeeze(dim=1)
        t_emb = self.text_embedding(tokenized_text_data)            # shape: (batch_size, max_seq) -> shape: (batch_size, max_seq, emb)
        t_emb = t_emb['last_hidden_state']
        
        t_emb = self.text_linear(t_emb) # shape: (batch_size, max_seq + 1, llm_hidden_size) -> shape: (batch_size, max_seq + 1, emb)
        
        try:
            x = self.tmp(t_emb).squeeze(dim=2)
            x = self.tmp2(x).squeeze(dim=1)
            return torch.nn.functional.sigmoid(x) * 4.0 + 1.0                     # shape: (batch_size, 1)
        except:
            breakpoint()


class UserTextModel(nn.Module):
    def __init__(self, args, max_seq_length, n_user_features, text_embedding_model) -> None:
        super(UserTextModel, self).__init__()
        self.user_embedding = nn.Linear(n_user_features, args.emb_size).to(args.device)
        
        self.text_embedding = text_embedding_model.to(args.device)
        self.llm_hidden_size = text_embedding_model.config.hidden_size
        self.text_linear = nn.Linear(self.llm_hidden_size, args.emb_size).to(args.device)
        
        self.mix = nn.Linear(max_seq_length + 1, 1).to(args.device)
        self.final = nn.Linear(args.emb_size, 1).to(args.device)

        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.elu = torch.nn.ELU()
        
    def forward(self, one_hot_data, tokenized_text_data):
        u_emb = self.user_embedding(one_hot_data)                          # shape: (batch_size, n_user_features) -> shape: (batch_size, emb)
        u_emb = u_emb.unsqueeze(dim = 1)                              # shape: (batch_size, emb) -> shape: (batch_size, 1, emb)
        u_emb = self.relu(u_emb)
        
        tokenized_text_data = tokenized_text_data.squeeze(dim=1)
        t_emb = self.text_embedding(tokenized_text_data)            # shape: (batch_size, max_seq) -> shape: (batch_size, max_seq, emb)
        t_emb = t_emb['last_hidden_state']
        
        t_emb = self.text_linear(t_emb) # shape: (batch_size, max_seq + 1, llm_hidden_size) -> shape: (batch_size, max_seq + 1, emb)
        
        try:
            x = torch.cat((u_emb, t_emb), dim=1)
            
            x = self.mix(x.permute(0,2,1))                                       # shape: (batch_size, max_seq + 1, emb) -> shape: (batch_size, 1, emb)
            x = self.relu(x)
            
            x = x.squeeze(dim=1)                                              # shape: (batch_size, 1, emb) -> shape: (batch_size, emb)
            x = self.final(x.permute(0,2,1))                                   # shape: (batch_size, emb) -> shape: (batch_size, 1)
            return torch.nn.functional.sigmoid(x) * 4.0 + 1.0                     # shape: (batch_size, 1)
        except:
            breakpoint()