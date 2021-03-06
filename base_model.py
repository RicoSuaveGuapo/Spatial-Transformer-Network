import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params


class BaseCnnModel(nn.Module):
    def __init__(self, input_length:int, input_ch:int=1, conv1_out_fea:int=32, conv2_out_fea:int=64, gap=False):
        super().__init__()
        
        self.input_length = input_length
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


        self.gap = nn.AdaptiveAvgPool2d(1) if gap else None
        self.cls = nn.Linear(self.conv2_out_fea, 10) if gap else nn.Linear(self.conv2_out_fea * (((input_length-9+1)//2-7+1)//2)**2, 10)


        #TODO the number of learnable parameter should be around 400,000
        #assert 399500 <= self.num_params() <= 400500, 'number of parameters should around 400k'

    def features(self, input):
        x = self.act(self.pool1(self.conv1(input))) # (B, self.conv1_out_fea, (input_length-9+1)//2, same)
        x = self.pool2(self.conv2(x))          # (B, self.conv2_out_fea, ((input_length-9+1)//2-7+1)//2, same)
        
        return x
    
    def logits(self, features):
        x = self.gap(features) if self.gap else features
        x = x.view(x.size(0), -1)
        x = self.cls(x)



        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)

        # since nn.CrossEntropy wiil do the softmax at first
        # we only need logit output here
        #x = F.softmax(x, dim=1)

        return x

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
        output = self.cls(self.act(self.fc2(output)))

        return output

    def num_params(self):
        return count_params(self)


class BaseStn(nn.Module):
    def __init__(self, model_name:str, input_ch:int, input_length:int, trans_type:str,
                conv1_kernel:int = 5, conv2_kernel:int = 5, conv1_outdim:int = 20,
                conv2_outdim:int = 20, theta_row:int=2, theta_col:int=3,
                fc1_outdim:int = 32, fc2_outdim:int = 32, fc3_outdim:int = 1, trans_task:str = 'Aff'
                ):
        """The base STN 

        Arguments:
            model_name {str} -- ST-CNN or ST-FCN
            trans_type {str} -- ['R', 'RTS', 'P', 'E', 'T', 'TU', None]
            input_ch {int} -- the input object channel
            input_length {int} -- the input object length

        Keyword Arguments:
            conv1_kernel {int} -- kernel size of convolution layer 1 (default: {5})
            conv2_kernel {int} -- kernel size of convolution layer 2 (default: {5})
            conv1_outdim {int} -- the output dim of convolution layer 1 (default: {20})
            conv2_outdim {int} -- the output dim of convolution layer 2 (default: {20})
            theta_row {int} -- the row count of parameters of the transformation (default: {2})
            theta_col {int} -- the col count of parameters of the transformation (default: {3})
           

            fc1_outdim {int} -- the output dim of fully connected layer 1 (default: {32})
            fc2_outdim {int} -- the output dim of fully connected layer 2 (default: {32})
            fc3_outdim {int} -- the output dim of fully connected layer 3 (default: {6})
            #TODO 

            trans_task {str} -- the type of transformation (default: {'Aff'})
        """


        assert model_name in ['ST-CNN', 'ST-FCN'], 'model name must be either ST-CNN or ST-FCN'
        assert trans_task in ['Aff','Proj','TPS'], 'trans_task must be one of Aff, Proj, TPS'
        assert trans_type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None], 'trans_type must be R, RTS, P, E, T, TU, None'
        
        super().__init__()
        self.model_name = model_name
        self.trans_task = trans_task
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
        self.theta = None
        

        if trans_type == 'R':
            self.fc_outdim = 1
            self.register_buffer('cos_matrix', torch.tensor([[1., 0, 0],
                                                         [0, 1., 0]], requires_grad=False).unsqueeze(0)) # (1,2,3)
            self.register_buffer('sin_matrix', torch.tensor([[0, -1., 0],
                                                         [1., 0, 0]], requires_grad=False).unsqueeze(0)) # (1,2,3)
        elif trans_type == 'RTS':
            self.fc_outdim = 6
        else:
            raise(TypeError)

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
            self.fc_loc = nn.Linear(self.conv_out_dim, self.fc_outdim)               # (self.fc_outdim)
            
        
        # For ST-FCN
        else:
            self.fc_loc = nn.Sequential(
                nn.Linear(self.input_ch*self.input_length**2, self.fc1_outdim),      # (32)
                nn.ReLU(),
                nn.Linear(self.fc1_outdim, self.fc2_outdim),                         # (32)
                nn.ReLU(),
                nn.Linear(self.fc2_outdim, self.fc3_outdim)                          # (self.fc_outdim)
                )


    def forward(self, input):
        if self.model_name == 'ST-CNN':
            output = self.conv_loc(input)
            output = output.view(output.size(0), -1)
            theta = self.fc_loc(output) #(N, self.fc_outdim)
            

            # 1. for only R transformation case
            if self.trans_type == 'R':
                theta = theta.unsqueeze(-1) # (N, 1, 1)
                            
                theta = torch.cos(theta) * self.cos_matrix + torch.sin(theta) * self.sin_matrix
            
            # 2. for general affine
            elif self.trans_type == 'RTS':
                theta = theta.view(-1, self.theta_row , self.theta_col) # (N, 2, 3)
                self.theta = theta

            else:
                #TODO
                pass


            # grid generator
            if self.trans_task == 'Aff':
                grid = F.affine_grid(theta, input.size(), align_corners=False)
                grid_sample = F.grid_sample(input, grid, align_corners=False, padding_mode="border", mode='bilinear')
            
                return grid_sample

            elif self.trans_task == 'Proj':
                #TODO
                pass
            else:
                #TODO
                pass

        else:
            theta = self.fc_loc(input.view(input.size(0),-1))
            # 1. for only R transformation case
            if self.trans_type == 'R':
                theta = theta.unsqueeze(-1) # (N, 1, 1)
                            
                theta = torch.cos(theta) * self.cos_matrix + torch.sin(theta) * self.sin_matrix
            
            # 2. for general affine
            elif self.trans_type == 'RTS':
                theta = theta.view(-1, self.theta_row , self.theta_col) # (n, 2, 3)

            else:
                #TODO
                pass

            # grid generator
            grid = F.affine_grid(theta, input.size(), align_corners=False)
            grid_sample = F.grid_sample(input, grid, align_corners=False, padding_mode="border", mode='bilinear')

            return grid_sample

    def num_params(self):
        return count_params(self)

    def gen_theta(self, input):
        if self.model_name == 'ST-CNN':
            output = self.conv_loc(input)
            output = output.view(output.size(0), -1)
            theta = self.fc_loc(output) #(N, self.fc_outdim)
            

            # 1. for only R transformation case
            if self.trans_type == 'R':
                theta = theta.unsqueeze(-1) # (N, 1, 1)
                            
                theta = torch.cos(theta) * self.cos_matrix + torch.sin(theta) * self.sin_matrix
            
            # 2. for general affine
            elif self.trans_type == 'RTS':
                theta = theta.view(-1, self.theta_row , self.theta_col) # (N, 2, 3)
                self.theta = theta
            
                return self.theta
            else:
                #TODO
                pass



        else:
            theta = self.fc_loc(input.view(input.size(0),-1))
            # 1. for only R transformation case
            if self.trans_type == 'R':
                theta = theta.unsqueeze(-1) # (N, 1, 1)
                            
                theta = torch.cos(theta) * self.cos_matrix + torch.sin(theta) * self.sin_matrix
                self.theta = theta
            
            # 2. for general affine
            elif self.trans_type == 'RTS':
                theta = theta.view(-1, self.theta_row , self.theta_col) # (n, 2, 3)
                self.theta = theta
            else:
                #TODO
                pass

            return self.theta

    


if __name__ == '__main__':
    #--test image
    # rand_img = torch.randn(1,1,28,28)

    # stn = BaseStn(model_name='ST-CNN', trans_task = 'Aff', trans_type = 'RTS',input_ch=rand_img.size(1) , input_length=rand_img.size(2))
    # out = stn(rand_img)
    # print("Output from stn:", out.size())
    
    # cnn = BaseCnnModel(input_length=rand_img.size(2), gap=True)
    # out = cnn(rand_img)
    # print("Output from CNN:", out.size())

    #fcn = BaseFcnModel(input_length=rand_img.size(2))
    #out = fcn(rand_img)
    #print("Output from FCN:", out.size())
    
    #--real image
    from torchvision.datasets import MNIST
    from matplotlib import pyplot as plt

    filepath = '/home/jarvis1121/AI/Rico_Repo/data'
    dataset = MNIST(root=filepath, train=False)
    # print(len(dataset)) # 60k for train, 10k for test
    
    idk = 25
    img, _ = dataset[idk]
    img_np = np.array(img)

    img = torch.from_numpy(img_np.reshape(1,1,28,28)).float()
    stn = BaseStn(model_name='ST-CNN', trans_task = 'Aff', trans_type = 'RTS',input_ch=1 , input_length=28)
    out = stn(img)
    print("Output from stn:", out.size())

    out_np = out.detach().numpy().reshape(28,28)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img_np, cmap='gray')
    axarr[1].imshow(out_np, cmap='gray')

    plt.show()
    
    #modules = stn.named_children()
    #for name, module in modules:
    #    if name == 'conv_loc':
     #       module.register_backward_hook(hook_fn_backward)


    #output = stn(rand_img)
    #output = output.
    #output.backward()

    #print(grad_in)
    # pass