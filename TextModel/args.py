import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    # emb_size: number of predictive factors
    parser.add_argument("--emb_size", default=32, type=int)
    parser.add_argument("--lr", type=float, default=0.001, help="lr")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--baseline", default=0, type=int)
    parser.add_argument("--dataset_path", default='dataset.pickle', type=str)
    parser.add_argument("--with_user", default=0, type=int)
    parser.add_argument("--save_path", default='/', type=str)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="hidden dropout p"
    )
    args = parser.parse_args()

    return args