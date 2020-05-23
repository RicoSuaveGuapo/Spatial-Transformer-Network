import argparse


import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import DistortedMNIST, MNISTAddition, CoLocalisationMNIST
from base_model import BaseCnnModel, BaseFcnModel, BaseStn
from model import CnnModel, FcnModel, StModel


#TODO 
# 1. tensorboard 視覺化 ST module 的梯度，目的是確保梯度有正常傳到 ST module，
# 2. 每訓練一段時間就秀出 transformed images
# 3. 可以在 train loop 加上 writer.add_images 秀出 transform 前後的比對圖

def build_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='The index of this experiment', type=int, default=1)
    parser.add_argument('--epoch', type=int, default = 1)

    parser.add_argument('--task_type', default='DistortedMNIST')
    parser.add_argument('--model_name', default='ST-CNN')
    parser.add_argument('--input_ch', default = 1)
    parser.add_argument('--input_length', default = 28)
    parser.add_argument('--transform_type', default='R')
    parser.add_argument('--val_split', type=float, default=1/6)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', default=0.01)
    return parser



def check_argparse(args):
    assert args.task_type in ['DistortedMNIST', 'MNISTAddition', 'CoLocalisationMNIST']
    assert args.transform_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
    assert args.model_name in ['CNN','FCN','ST-CNN','ST-FCN']



def build_train_val_test_dataset(args):
    if args.task_type == 'DistortedMNIST':
        train_dataset = DistortedMNIST(mode='train', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        val_dataset   = DistortedMNIST(mode='val', transform_type=args.transform_type, val_split=args.val_split, seed=args.seed)
        test_dataset   = DistortedMNIST(mode='test', transform_type=args.transform_type, seed=args.seed)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader

    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass
def build_scheduler(optimizer):
    lambdaAll = lambda iteration: 0.1 ** (iteration//50000)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambdaAll)
    return scheduler

  


def main():
    # device
    assert torch.cuda.is_available(), 'It is better to train with GPU'
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # data
    print('\n-------- Data Preparing --------\n')

    train_dataloader, val_dataloader, test_dataloader = build_train_val_test_dataset(args)

    print('\n-------- Data Preparing Done! --------\n')


    print('\n-------- Preparing Model --------\n')
    # model
    if args.task_type == 'DistortedMNIST':
        if args.model_name == 'ST-CNN':            
            stn = BaseStn(model_name=args.model_name, input_ch=args.input_ch , input_length=args.input_length)
            base_cnn = BaseCnnModel(input_length=args.input_length)
            model = StModel(base_stn = stn, base_nn_model = base_cnn)

            # pass to CUDA device
            model = model.to(device)
            

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = build_scheduler(optimizer)
            

            # callbacks
            #filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
            # https://hackmd.io/l9Kcq74ARcC_qimz5QRM3g
        
        elif args.model_name == 'ST-FCN':
            stn = BaseStn(model_name=args.model_name, input_ch=args.input_ch , input_length=args.input_length)
            base_fcn = BaseFcnModel(input_length=args.input_length)
            model = StModel(base_stn = stn, base_nn_model = base_fcn)

            # pass to CUDA device
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            # pass to CUDA device
            #criterion.to(device)

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = build_scheduler(optimizer)

            # callbacks
            #filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
            
        
        elif args.model_name == 'CNN':
            model = CnnModel()
            
            # pass to CUDA device
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            # pass to CUDA device
            #criterion.to(device)

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = build_scheduler(optimizer)

            
            # callbacks
            #filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'
        else:
            model = FcnModel()
            
            # pass to CUDA device
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            # pass to CUDA device
            #criterion.to(device)

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = build_scheduler(optimizer)


            # callbacks
            #filepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/callback/trial_{int(args.exp)}'            
    
    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass
    

    print('\n-------- Preparing Model Done! --------\n')

    

    # train
    print('\n-------- Starting Training --------\n')
    # prepare the tensorboard
    writer = SummaryWriter(f'runs/trial_{args.exp}')

    for epoch in range(args.epoch): #TODO paper use 150*1000 iterations ~ 769 epoch in batch_size = 256
        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        # TRAINING LOOP
        print('---Training Loop begins---')
        for i, data in enumerate(train_dataloader, start=0): 
            # move CUDA device
            input, target = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
            writer.add_scalar('Averaged loss', loss.item(), 196*epoch + i)
            writer.add_scalar('ST Gradients Norm', model.norm, 196*epoch + i)
            if i % 20 == 19:
                print(
                    f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss/20)
                )
                train_running_loss = 0.0
            elif i == 195:
                print(
                    f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss/16)
                )
        print('---Training Loop ends---')
        
        # catch the transformed image though ST, after one epoch
        with torch.no_grad():
            # number of images to show
            n = 6
            origi_img = input[:n,...].clone().detach() #(4, C, H, W)
            trans_img = stn(origi_img) #(4, C, H, W)
            img = torch.cat((origi_img,trans_img), dim=0) #(4+4, C, H, W)
            img = make_grid(img, nrow=n)
            writer.add_image(f"Original-Up, ST-Down images in epoch_{epoch+1}", img)
        
        # VALIDATION LOOP
        with torch.no_grad():
            val_run_loss = 0.0
            print('---Validaion Loop begins---')
            batch_count = 0
            total_count = 0
            correct_count = 0
            for i, data in enumerate(val_dataloader, start=0):
                input, target = data[0].to(device), data[1].to(device)

                output = model(input)
                loss = criterion(output, target)

                _, predicted = torch.max(output, 1)

                val_run_loss += loss.item()
                batch_count += 1
                total_count += target.size(0)

                correct_count += (predicted == target).sum().item()
            
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count
            
            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"Loss of {epoch+1} epoch is %.3f" % (val_run_loss))
            print(f"Accuracy is {accuracy} %")
                
            print('---Validaion Loop ends---')
    writer.close()
    print('\n-------- End Training --------\n')
    

    print('\n-------- Saving Model --------\n')

    savepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/model_save/{str(args.exp)}_{str(args.task_type)}_{str(args.transform_type)}_{str(args.model_name)}.pth'
    torch.save(model.state_dict(), savepath)
        
    print('\n-------- Saved --------\n')
    print(f'\n== Trial {args.exp} finished ==\n')

#----------------------
# 10 epoch ~ 1m, accuracy 86.6%
#----------------------
if __name__ == '__main__':
    main()