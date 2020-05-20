import argparse


import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from datasets import DistortedMNIST, MNISTAddition, CoLocalisationMNIST
from base_model import BaseCnnModel, BaseFcnModel, BaseStn
from model import CnnModel, FcnModel, StModel


def build_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='The index of this experiment', default=None)

    parser.add_argument('--task_type', default='DistortedMNIST')
    parser.add_argument('--model_name', default='ST-CNN')
    parser.add_argument('--input_ch', default = 1)
    parser.add_argument('--input_length', default = 28)
    parser.add_argument('--transform_type', default=None)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', default=0.01)
    return parser



def check_argparse(args):
    assert args.task_type in ['DistortedMNIST', 'MNISTAddition', 'CoLocalisationMNIST']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
    assert args.model_name in ['CNN','FCN','ST-CNN','ST-FCN']



def build_train_val_dataset(args):
    if args.task_type == 'DistortedMNIST':
        train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        val_dataset   = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, val_dataloader

    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass
def build_scheduler(optimizer):
    lambdaAll = lambda epoch: 0.1 * (epoch//50000)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambdaAll)
    return scheduler


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
    if args.task_type == 'DistortedMNIST':
        if args.model_name == 'ST-CNN':            
            stn = BaseStn(model_name=args.model_name, input_ch=args.input_ch , input_length=args.input_length)
            base_cnn = BaseCnnModel(input_length=args.input_length)
            model = StModel(base_stn = stn, base_nn_model = base_cnn)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = build_scheduler(optimizer)
            

            # callbacks
            filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
            # https://hackmd.io/l9Kcq74ARcC_qimz5QRM3g
        
        elif args.model_name == 'ST-FCN':
            stn = BaseStn(model_name=args.model_name, input_ch=args.input_ch , input_length=args.input_length)
            base_fcn = BaseFcnModel(input_length=args.input_length)
            model = StModel(base_stn = stn, base_nn_model = base_fcn)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = 

            # callbacks
            filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
            
        
        elif args.model_name == 'CNN':
            model = CnnModel()
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = 

            
            # callbacks
            filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
        else:
            model = FcnModel()
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = 


            # callbacks
            filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'            
    
    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass


    print('\n-------- Model Bulided --------\n')

    

    # train
    print('\n-------- Starting Training --------\n')
    for input, target in train_dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(input, target)
        optimizer.step()
        scheduler.step()

    print('\n-------- End Training --------\n')


if __name__ == '__main__':
    main()