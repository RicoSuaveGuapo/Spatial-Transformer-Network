import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import DistortedMNIST, MNISTAddition, CoLocalisationMNIST
from base_model import BaseCnnModel, BaseFcnModel, BaseStn


from base_model import BaseStn
from train import build_train_val_test_dataset, build_argparse, check_argparse



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # data 
    train_dataloader, val_dataloader, _ = build_train_val_test_dataset(args)

    # model
    if args.task_type == 'DistortedMNIST':
        if args.model_name == 'ST-CNN':            
            model = BaseStn(model_name=args.model_name, trans_type=args.trans_type, input_ch=args.input_ch , input_length=args.input_length)
            
            # pass to CUDA device
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
                        
        
        elif args.model_name == 'ST-FCN':
            model = BaseStn(model_name=args.model_name, trans_type=args.trans_type, input_ch=args.input_ch , input_length=args.input_length)
            
            # pass to CUDA device
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

    elif args.task_type == 'MNISTAddition':
        #TODO
        pass

    else:
        #TODO
        pass
    
    # training
    writer = SummaryWriter(f'runs/trial_stn_{args.exp}')
    

    for epoch in range(args.epoch):
        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        # TRAINING LOOP
        print('---Training Loop begins---')
        for i, data in enumerate(train_dataloader, start=0): 
            # move CUDA device
            input = data[0].to(device)
            target_theta = torch.tensor([[1,0,0],[0,1,0]], requires_grad=False, dtype=torch.float)
            target_theta = target_theta.unsqueeze(0)
            target_theta = target_theta.expand(len(input), 2, 3).to(device)
                        
            optimizer.zero_grad()
            output = model.gen_theta(input)
            loss = criterion(output, target_theta)
            output_average = torch.mean(output, dim=0)

            if loss <=0.02:
                print(f'iteration: {i}')
                print(
                    f'theta average: {output_average}'
                )
                break
            else:
                pass

            loss.backward()
            optimizer.step()
            

            train_running_loss += loss.item()
            
            writer.add_scalar('Averaged loss', loss.item(), 196*epoch + i)
            
            if i % 20 == 19:
                print(
                    f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss/20)
                )
                print(
                    f'theta average: {output_average}'
                )
                train_running_loss = 0.0
            elif i == 195:
                print(
                    f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss/16)
                )
                print(
                    f'theta average: {output_average}'
                )
        print('---Training Loop ends---')
        
        # catch the transformed image though ST, after one epoch
        with torch.no_grad():
            # number of images to show
            n = 6
            origi_img = input[:n,...].clone().detach() #(4, C, H, W)
            trans_img = model(origi_img) #(4, C, H, W)
            img = torch.cat((origi_img,trans_img), dim=0) #(4+4, C, H, W)
            img = make_grid(img, nrow=n)
            writer.add_image(f"Original-Up, ST-Down images in epoch_{epoch+1}", img)
        
        # VALIDATION LOOP
        with torch.no_grad():
            val_run_loss = 0.0
            print('---Validaion Loop begins---')
            batch_count = 0
            
            for i, data in enumerate(val_dataloader, start=0):
                input = data[0].to(device)
                target_theta = torch.tensor([[1,0,0],[0,1,0]], requires_grad=False, dtype=torch.float)
                target_theta = target_theta.unsqueeze(0)
                target_theta = target_theta.expand(len(input), 2, 3).to(device)

                output = model.gen_theta(input)
                loss = criterion(output, target_theta)

                val_run_loss += loss.item()
                batch_count += 1
                            
            val_run_loss = val_run_loss/batch_count
            
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"Loss of {epoch+1} epoch is %.3f" % (val_run_loss))
                
            print('---Validaion Loop ends---')
    writer.close()
    savepath = f'/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/model_save/stn_{str(args.exp)}_{str(args.task_type)}_{str(args.trans_type)}_{str(args.model_name)}.pth'
    torch.save(model.state_dict(), savepath)

if __name__ == '__main__':
    main()
    # import numpy as np
    # model = BaseStn(model_name='ST-CNN', trans_type='RTS', input_ch=1 , input_length=42)
    # model.load_state_dict(torch.load('/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/model_save/stn_7_DistortedMNIST_RTS_ST-CNN.pth'))
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, torch.min(torch.abs(param.data)))