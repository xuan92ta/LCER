import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generating counterfactual explanations for Mult-VAE.")

    parser.add_argument('--gpu_id', type=int, default=7, help='ID of GPU.')
    
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m', 'alishop', 'epinions'], help='Name of dataset.')
    parser.add_argument('--processed_dir', type=str, default='pro_sg/', help='Directory of preprocessed dataset.')
    parser.add_argument('--checkpoint_dir', type=str, default='chkpt_vae/', help='Directory of checkpoint.')
    
    parser.add_argument('--lam', type=float, default=0.01, help='The expected scale of loss of length.')
    parser.add_argument('--alpha', type=float, default=0.1, help='The slack variable of hinge loss.')
    parser.add_argument('--k', type=int, default=5, help='The number of recommended items for each user.')
    parser.add_argument('--seed', type=int, default=2022, help='The random number seed.')
    parser.add_argument('--epochs', type=int, default=3000, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate.')

    return parser.parse_args()