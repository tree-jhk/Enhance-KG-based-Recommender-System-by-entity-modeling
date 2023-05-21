import torch
from utils import *
from model import *
from tqdm import tqdm
import csv

dic = {0:"TextModel", 1:"UserTextModel"}

def train(args, train_dataloader, valid_dataloader, model, optimizer):
    best_loss = 1e9
    train_result = pd.DataFrame()
    valid_result = pd.DataFrame()
    
    os.makedirs("result/", exist_ok=True)
    with open('result/train_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Valid Loss'])
        for epoch in range(args.num_epochs):
            model.train()
            losses = 0
            train_y_list, train_y_hat_list = list(), list()
            for i, batch_data in tqdm(enumerate(train_dataloader)):
                one_hot_data, tokenized_text_data, y = batch_data
                optimizer.zero_grad()
                y_hat = model(one_hot_data.to(args.device), tokenized_text_data.to(args.device))
                train_loss = torch.nn.functional.mse_loss(y, y_hat).to('cpu')
                train_loss.backward()
                optimizer.step()
                losses += train_loss.item()
                
                # for result
                train_y_list.extend(y.tolist())
                train_y_hat_list.extend(y_hat.tolist())
                
            train_loss_ = losses / (i + 1)
            print(f'Epoch: {epoch} Avg Loss: {train_loss_ }')
            valid_loss, valid_y_list, valid_y_hat_list = valid(args, valid_dataloader, model)
            if best_loss > valid_loss:
                best_loss = valid_loss
                print(f'Saving model with valid loss {valid_loss:.4f}...')
                os.makedirs("model", exist_ok=True)
                save_path = "model" + args.save_path + dic[args.with_user] + f'model_{0}.pt'
                torch.save(model.state_dict(), save_path)
                
                
                train_y_list = torch.Tensor(train_y_list)
                train_y_hat_list = torch.Tensor(train_y_hat_list)
                valid_y_list = torch.Tensor(valid_y_list)
                valid_y_hat_list = torch.Tensor(valid_y_hat_list)
                
                round_float = lambda x: [round(x[0], 3), round(x[1], 3)]
                
                train_result = list(map(round_float, torch.stack([train_y_list, train_y_hat_list], dim=1).tolist()))
                valid_result = list(map(round_float, torch.stack([valid_y_list, valid_y_hat_list], dim=1).tolist()))
                
                train_result = pd.DataFrame(train_result, columns=['GT', 'Predict'])
                valid_result = pd.DataFrame(valid_result, columns=['GT', 'Predict'])
                
                os.makedirs("result", exist_ok=True)
                train_result.to_csv("result/train_result.csv", index=False)
                valid_result.to_csv("result/valid_result.csv", index=False)
                
            row = [epoch, train_loss_, valid_loss]
            writer.writerow(row)


def valid(args, valid_dataloader, model):
    model.eval()
    losses = 0
    valid_y_list, valid_y_hat_list = list(), list()
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(valid_dataloader)):
            one_hot_data, tokenized_text_data, y = batch_data
            y_hat = model(one_hot_data.to(args.device), tokenized_text_data.to(args.device))
            valid_loss = torch.nn.functional.mse_loss(y, y_hat).to('cpu')
            losses += valid_loss.item()
            
            # for result
            valid_y_list.extend(y.tolist())
            valid_y_hat_list.extend(y_hat.tolist())
            
        valid_loss_ = losses / (i + 1)
        print(f'Validation Avg Loss: {valid_loss_}')
    return valid_loss_, valid_y_list, valid_y_hat_list