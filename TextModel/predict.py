import torch
from utils import *
from model import *
from tqdm import tqdm
from train import valid
from pickle import Unpickler
from preprocess import *
from dataset import *
from args import *

dic = {0:"TextModel", 1:"UserTextModel"}

def predict(args, predict_dataloader, model, original_data):
    predict_result = pd.DataFrame()
    print(f'{len(predict_dataloader) =}')
    print(f'{args.batch_size =}')
    print(f'{len(predict_dataloader) // (args.batch_size) + 1 =}')
    _, predict_y_list, predict_y_hat_list = valid(args, predict_dataloader, model)
    predict_y_list = torch.Tensor(predict_y_list)
    predict_y_hat_list = torch.Tensor(predict_y_hat_list)
    
    round_float = lambda x: [round(x[0], 3), round(x[1], 3)]
    
    predict_result = list(map(round_float, torch.stack([predict_y_list, predict_y_hat_list], dim=1).tolist()))
    
    predict_result = pd.DataFrame(predict_result, columns=['GT', 'Predict'])
    
    result = pd.concat([predict_result, original_data], axis=1)
    result = result[['rating', 'Predict', 'user_id', 'movie_id', 'title']]
    
    os.makedirs("result", exist_ok=True)
    result.to_csv("result/final_predict_result.csv", index=False)

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
    original_data, predict_data = dataset.original_data, dataset.predict
    
    predict_dataset = UserTextDataset(args, predict_data)
    predict_dataloader = DataLoader(predict_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
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

    model.load_state_dict(torch.load("model/TextModel_0.pt", map_location=args.device))
    
    predict(args, predict_dataloader, model, original_data)