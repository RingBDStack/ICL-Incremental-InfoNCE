import argparse
import torch
import os
from dataset import process_data
from encoder import build_encoder
from train import train
from train_ICL import train_ICL
from train_ICL_RL import train_ICL_RL
from train_ICL_MAML import train_ICL_MAML
from train_ICL_MAML_RL import train_ICL_MAML_RL

def init_para():
    parser = argparse.ArgumentParser(description="ICL")
       
    # dataset
    parser.add_argument('-m', '--mode', default='cv', type=str, help="cv or graph.")
    parser.add_argument('-d', '--dataset',default='ImageNet-2', type=str, help="Dataset.")
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help="delta_N / (N + delta_N).")
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help="Batch size.")
    
    # model
    parser.add_argument('-en', '--encoder', default='resnet18', type=str, help="Encoder.")
    parser.add_argument('-dim', '--dim', default=64, type=int, help="Dimension of embeddings.")
    parser.add_argument('-l', '--layer_num', default=2, type=int, help="Number of layers.")
    
    # train    
    parser.add_argument('-e', '--epochs', default=1000, type=int, help='Training epochs.')
    parser.add_argument('-p', '--patience', default=20, type=int, help='Patient epochs to wait before early stopping.')
    parser.add_argument('-lr', '--lr', default=1e-5, type=float, help='Learning rate.')
    parser.add_argument('-w', '--weight_decay', default=1e-5, type=float, help='Weight decay.')
       
    # RL    
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--hidden', default=10, type=int, help='hidden num of actor-critic layer')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
         
    
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    
    return args

   
def main():
    args = init_para()
    data, old_data, new_data, n_feature, n_class = process_data(args.mode, args.dataset, args.alpha)
    print(n_feature, n_class)
    print(args.alpha)
    args.n_class = n_class
        
    # train the old data, and save the model
    print("Inference!")
    encoder = build_encoder(args.mode, args.encoder, n_feature, args.dim//args.layer_num, args.layer_num)
    pkl_save, loss0 = train(data, old_data, new_data, encoder, args, 'old', True)
               
    # train the new data using ICL-MAML-RL
    print("ICL-MAML-RL!")
    encoder.load_state_dict(torch.load(pkl_save))
    train_ICL_MAML_RL(data, old_data, new_data, encoder, args, loss0)
    
    # train the new data using ICL-RL
    print("ICL-RL!")
    encoder.load_state_dict(torch.load(pkl_save))
    train_ICL_RL(data, old_data, new_data, encoder, args, loss0)
    
    # train the new data using ICL-MAML
    print("ICL-MAML!")
    encoder.load_state_dict(torch.load(pkl_save))
    train_ICL_MAML(data, old_data, new_data, encoder, args)
        
    # train the new data using ICL
    print("ICL!")
    encoder.load_state_dict(torch.load(pkl_save))
    train_ICL(data, old_data, new_data, encoder, args)
    
    os.remove(pkl_save)
    
    # train all of the data
    print("All!")
    encoder = build_encoder(args.mode, args.encoder, n_feature, args.dim//args.layer_num, args.layer_num).to(args.device)
    train(data, old_data, new_data, encoder, args, 'all')
     
    
    
if __name__ == "__main__":
    print('Begin!')
    main()
    print('Done!')
