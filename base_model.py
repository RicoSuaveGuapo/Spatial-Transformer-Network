import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params


class BaseCnnModel(nn.Module):
    def __init__(self, input_length:int, input_ch:int=1, conv1_out_fea:int=32, conv2_out_fea:int=64):
        super().__init__()
        
        # the number of filters shold be around 32~64
        assert 32<= conv1_out_fea <= 64, 'number of filters in conv1 must in the range of 32~64'
        assert 32<= conv2_out_fea <= 64, 'number of filters in conv1 must in the range of 32~64'
        
        self.conv1_out_fea = conv1_out_fea
        self.conv2_out_fea = conv2_out_fea

        self.conv1 = nn.Conv2d(input_ch, self.conv1_out_fea, 9) # (self.conv1_out_fea, input_length-9+1, same)
        self.pool1 = nn.MaxPool2d(2) # (self.conv1_out_fea, (input_length-9+1)//2, same)
        self.act   = nn.ReLU()
        self.conv2 = nn.Conv2d(self.conv1_out_fea, self.conv2_out_fea, 7) # (self.conv2_out_fea, (input_length-9+1)//2-7+1, same)
        self.pool2 = nn.MaxPool2d(2) # (self.conv2_out_fea, ((input_length-9+1)//2-7+1)//2, same)
        self.cls = nn.Linear(self.conv2_out_fea * (((input_length-9+1)//2-7+1)//2)**2, 10)

        #TODO the number of learnable parameter should be around 400,000
        #assert 399500 <= self.num_params() <= 400500, 'number of parameters should around 400k'
    
    def forward(self, input):
        output = self.act(self.pool1(self.conv1(input)))
        output = self.pool2(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.cls(output)

        output = F.softmax(output, dim=1)

        return output

    def num_params(self):
        return count_params(self)


class BaseFcnModel(nn.Module):
    def __init__(self, input_length:int, input_ch:int=1, fc1_unit:int=128, fc2_unit:int=256):
        super().__init__()
        # the number of unit should be around 128~256
        assert 128 <= fc1_unit <= 256, 'number of unit in fc1 should around 128~256'
        assert 128 <= fc2_unit <= 256, 'number of unit in fc2 should around 128~256'

        self.fc1_unit = fc1_unit
        self.fc2_unit = fc2_unit
        
        self.fc1 = nn.Linear(input_ch*input_length*input_length, self.fc1_unit)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_unit, self.fc2_unit)
        self.cls = nn.Linear(self.fc2_unit, 10)

        #TODO
        #assert 399500 <= self.num_params() <= 400500, 'number of parameters should around 400k'

    def forward(self, input):

        input = input.view(input.size(0), -1)
        output = self.act(self.fc1(input))
        output = self.cls(self.fc2(output))

        output = F.softmax(output, dim=1)
        return output

    def num_params(self):
        return count_params(self)


class BaseStn(nn.Module):
    def __init__(self, model_name:str, input_ch:int, input_length:int,
                conv1_kernel:int = 5, conv2_kernel:int = 5, conv1_outdim:int = 20,
                conv2_outdim:int = 20, theta_row:int=2, theta_col:int=3, fc_outdim:int=1,
                fc1_outdim:int = 32, fc2_outdim:int = 32, fc3_outdim:int = 1, trans_type:str = 'Aff'
                ):
        """The base STN 

        Arguments:
            model_name {str} -- ST-CNN or ST-FCN
            input_ch {int} -- the input object channel
            input_length {int} -- the input object length

        Keyword Arguments:
            conv1_kernel {int} -- kernel size of convolution layer 1 (default: {5})
            conv2_kernel {int} -- kernel size of convolution layer 2 (default: {5})
            conv1_outdim {int} -- the output dim of convolution layer 1 (default: {20})
            conv2_outdim {int} -- the output dim of convolution layer 2 (default: {20})
            theta_row {int} -- the row count of parameters of the transformation (default: {2})
            theta_col {int} -- the col count of parameters of the transformation (default: {3})
            fc_outdim {int} -- the output dim of the last layer in ST for ST-CNN (default: {6})
            #TODO note that fc_outdim for affine transformation is 6, however for more advance 
            transformation must need 20 parameters.

            fc1_outdim {int} -- the output dim of fully connected layer 1 (default: {32})
            fc2_outdim {int} -- the output dim of fully connected layer 2 (default: {32})
            fc3_outdim {int} -- the output dim of fully connected layer 3 (default: {6})
            #TODO 

            trans_type {str} -- the type of transformation (default: {'Aff'})
        """


        assert model_name in ['ST-CNN', 'ST-FCN'], "model name must be either ST-CNN or ST-FCN"
        assert trans_type in ['Aff','Proj','TPS'], 'transformation_type must be one of Aff, Proj, TPS'
        
        super().__init__()
        self.model_name = model_name
        self.trans_type = trans_type

        self.input_ch = input_ch
        self.input_length= input_length
        self.conv1_kernel = conv1_kernel
        self.conv2_kernel = conv2_kernel
        self.conv1_outdim = conv1_outdim
        self.conv2_outdim = conv2_outdim 

        self.conv_out_dim = self.conv2_outdim*((((self.input_length - self.conv1_kernel)+1)//2 - 
                            self.conv2_kernel)+1)**2
        self.theta_row = theta_row
        self.theta_col = theta_col
        self.fc_outdim = fc_outdim

        self.fc1_outdim = fc1_outdim
        self.fc2_outdim = fc2_outdim
        self.fc3_outdim = fc3_outdim

        # --localisation networks --
        # For ST-CNN
        if model_name == 'ST-CNN':
            self.conv_loc = nn.Sequential(
                nn.Conv2d(self.input_ch, self.conv1_outdim, self.conv1_kernel),     # (20, 24, 24)
                nn.MaxPool2d(2),                                                    # (20, 12, 12)
                nn.ReLU(),
                nn.Conv2d(self.conv1_outdim, self.conv2_outdim, self.conv2_kernel), # (20, 8, 8)
                nn.ReLU()
                )
            self.fc_loc = nn.Linear(self.conv_out_dim, self.fc_outdim)               # (6)
        
        # For ST-FCN
        else:
            self.fc_loc = nn.Sequential(
                nn.Linear(self.input_ch*self.input_length**2, self.fc1_outdim),         # (32)
                nn.ReLU(),
                nn.Linear(self.fc1_outdim, self.fc2_outdim),                         # (32)
                nn.ReLU(),
                nn.Linear(self.fc2_outdim, self.fc3_outdim)                          # (6)
                )


    def forward(self, input):
        if self.model_name == 'ST-CNN':
            output = self.conv_loc(input)
            output = output.view(output.size(0), -1)
            theta = self.fc_loc(output) #(N, self.fc_outdim)
            
            # 1. for general affine
            #theta = theta.view(-1, self.theta_row , self.theta_col)

            # 2. for only R transformation case
            theta = theta.unsqueeze(-1) # (N, 1, 1)
            cos_matrix = torch.tensor([[1., 0, 0],
                                        [0, 1., 0]], requires_grad=False) # (2,3)
            sin_matrix = torch.tensor([[0, -1., 0],
                                        [1., 0, 0]], requires_grad=False) # (2,3)
            
            cos_matrix = cos_matrix.unsqueeze(0) # (1,2,3)
            sin_matrix = sin_matrix.unsqueeze(0) # (1,2,3)
            theta = torch.cos(theta) * cos_matrix + torch.sin(theta) * sin_matrix
            
            # grid generator
            if self.trans_type == 'Aff':
                grid = F.affine_grid(theta, input.size(), align_corners=False)
                grid_sample = F.grid_sample(input, grid, align_corners=False, padding_mode="border", mode='bilinear')
            
                return grid_sample

            elif self.trans_type == 'Proj':
                #TODO
                pass
            else:
                #TODO
                pass

        else:
            theta = self.fc_loc(input.view(input.size(0),-1))
            # 1. for general affine
            #theta = theta.view(-1, self.theta_row , self.theta_col)

            # 2. for only R transformation case
            theta = theta.unsqueeze(-1) # (N, 1, 1)
            cos_matrix = torch.tensor([[1., 0, 0],
                                        [0, 1., 0]], requires_grad=False) # (2,3)
            sin_matrix = torch.tensor([[0, -1., 0],
                                        [1., 0, 0]], requires_grad=False) # (2,3)
            
            cos_matrix = cos_matrix.unsqueeze(0) # (1,2,3)
            sin_matrix = sin_matrix.unsqueeze(0) # (1,2,3)
            theta = torch.cos(theta) * cos_matrix + torch.sin(theta) * sin_matrix
            
            
            # grid generator
            grid = F.affine_grid(theta, input.size(), align_corners=False)
            grid_sample = F.grid_sample(input, grid, align_corners=False, padding_mode="border", mode='bilinear')

            return grid_sample

    def num_params(self):
        return count_params(self)

if __name__ == '__main__':
    #--test image
    #rand_img = torch.randn(1,1,28,28)
    #print(rand_img)
    #stn = BaseStn(model_name='ST-CNN', input_ch=rand_img.size(1) , input_length=rand_img.size(2))
    #out = stn(rand_img)
    #print("Output from stn:", out.size())
    
    #cnn = BaseCnnModel(input_length=rand_img.size(2))
    #out = cnn(rand_img)
    #print("Output from CNN:", out.size())

    #fcn = BaseFcnModel(input_length=rand_img.size(2))
    #out = fcn(rand_img)
    #print("Output from FCN:", out.size())
    
    #--real image
    from torchvision.datasets import MNIST
    from matplotlib import pyplot as plt

    filepath = '/home/jarvis1121/AI/Rico_Repo/data'
    dataset = MNIST(root=filepath, train=True)
    idk = 5
    img, _ = dataset[idk]
    img_np = np.array(img)
    img = torch.from_numpy(img_np.reshape(1,1,28,28)).float()
    stn = BaseStn(model_name='ST-CNN', input_ch=img.size(1) , input_length=img.size(2))
    out = stn(img)
    print("Output from stn:", out.size())

    out_np = out.detach().numpy().reshape(28,28)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img_np, cmap='gray')
    axarr[1].imshow(out_np, cmap='gray')

    plt.show()
    
