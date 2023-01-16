import os
import sys
import argparse
import torch

SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-2])
sys.path.append(os.path.dirname(SCRIPT_DIR))

from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For the 1-dim setup with symm constraint, look at the lines along beta_vae
    # Just use beta_vae

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--seq_len', type=int, default=15)
    parser.add_argument('--data_folder', default='cleanTrain')
    parser.add_argument('--no_rnn', action='store_true')
    parser.add_argument('--no_symm', action='store_true')
    parser.add_argument('--no_rep', action='store_true')
    parser.add_argument('--symm_against_rnn', action='store_true')
    parser.add_argument('--additional_symm_steps', type=int, default=0) 
    parser.add_argument('--symm_start_step', type=int, default=0) # Set this to 15 to apply symm loss only on OOR steps 

    # RNN params
    parser.add_argument('--rnn_num_layers', type=int, default=1)
    parser.add_argument('--rnn_hidden_size', type=int, default=256)
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--beta_vae', action='store_true')
    parser.add_argument('--ae', action='store_true')

    # Hyper params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z_rnn_loss_scalar', type=float, default=2)

    parser.add_argument('--n_runs', type=int, default=1)

    # Eval
    parser.add_argument('--eval_recons', action='store_true')

    args = parser.parse_args()

    CONFIG['seq_len'] = args.seq_len
    CONFIG['train_data_path'] = f'../../data/{args.data_folder}'
    CONFIG['no_rnn'] = args.no_rnn
    CONFIG['no_symm'] = args.no_symm
    CONFIG['no_repetition'] = args.no_rep
    CONFIG['additional_symm_steps'] = args.additional_symm_steps
    CONFIG['symm_start_step'] = args.symm_start_step    
    CONFIG['symm_against_rnn'] = args.symm_against_rnn
    CONFIG['rnn_num_layers'] = args.rnn_num_layers
    CONFIG['rnn_hidden_size'] = args.rnn_hidden_size
    CONFIG['GRU'] = args.gru
    CONFIG['learning_rate'] = args.lr
    CONFIG['z_rnn_loss_scalar'] = args.z_rnn_loss_scalar
    CONFIG['beta_vae'] = args.beta_vae
    CONFIG['ae'] = args.ae
    CONFIG['eval_recons'] = args.eval_recons
    print(CONFIG['eval_recons'])


    # torch.manual_seed(21)

    # Loop for multiple runs
    for i in range(args.n_runs):
        name = args.name + '_' + str(i)
        CONFIG['name'] = name
        CONFIG['model_path'] =  f'{name}_Conv2dNOTGruConv2d_symmetry.pt'
        CONFIG['train_result_path'] = f'./new_dumpster/{name}TrainingResults/'
        CONFIG['train_record_path'] = f'./new_dumpster/{name}Train_record.txt'
        CONFIG['eval_record_path'] = f'./new_dumpster/{name}Eval_record.txt'

        trainer = BallTrainer(CONFIG)
        if args.eval_recons or is_need_train(CONFIG):
            self_recon, pred_recon, rnn_prior = trainer.train()
            for m in [self_recon, pred_recon, rnn_prior]:
                m = torch.tensor(m)
                print(torch.mean(m))
