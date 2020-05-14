import argparse


import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from datasets import DistortedMNIST, MNISTAddition, CoLocalisationMNIST


def build_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='The index of this experiment', default=None)

    parser.add_argument('--task_type', default='DistortedMNIST')
    parser.add_argument('--transform_type', default=None)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=32)
    return parser



def check_argparse(args):
    assert args.task_type in ['DistortedMNIST', 'MNISTAddition', 'CoLocalisationMNIST']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]



def build_train_val_dataset(args):
    if args.task_type == 'DistortedMNIST':
        train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type,val_split=args.val_split, seed=args.seed)
        val_dataset   = DistortedMNIST(mode='val', transform_type=args.transform_type,val_split=args.val_split)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, val_dataloader

    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass

def main():

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # data
    print('\n-------- Data Preparing --------\n')

    train_dataloader, val_dataloader = build_train_val_dataset(args)

    print('\n-------- Data Preparing Done! --------\n')

    print('\n-------- Preparing Model --------\n')
    
    # model
    criterion = 
    cnn_model =


    optimizer = 
    scheduler = 

    st_cnn_model =

    # callbacks
    filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
    # https://hackmd.io/l9Kcq74ARcC_qimz5QRM3g


    print('\n-------- Model Bulided --------\n')

    # train
    print('\n-------- Starting Training --------\n')

    print('\n-------- End Training --------\n')


if __name__ == '__main__':
    main()