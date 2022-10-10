import os
import sys
import argparse

SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
sys.path.append(os.path.dirname(SCRIPT_DIR))

from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--data_folder')

    args = parser.parse_args()

    NAME = args.name
    CONFIG['seq_len'] = args.seq_len
    CONFIG['train_data_path'] = f'../../data/{args.data_folder}'

    trainer = BallTrainer(CONFIG)
    if is_need_train(CONFIG):
        trainer.train()
