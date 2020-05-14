import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_cnn_model(nn.Module):
    #TODO the number of filiters are b/t 32~64
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 9) #(32, 20, 20) 
        self.pool1 = nn.MaxPool2d(2)    #(32, 10, 10)
        self.act   = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64,7) #(64, 4, 4)
        self.pool2 = nn.MaxPool2d(2)    #(64, 2, 2)
        self.cls = nn.Linear(64*2*2, 10)

    
    def forward(self, input):

        assert max(input.size()) == 28,"input img size should be 28x28"

        output = self.act(self.pool1(self.conv1(input)))
        output = self.pool2(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.cls(output)

        output = F.softmax(output, dim=1)

        return output

class Base_fcn_model(nn.Module):
    #TODO the number of units per layer are 128~256
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.cls = nn.Linear(256, 10)

    def forward(self, input):

        assert max(input.size()) == 28,"input img size should be 28x28"

        input = input.view(input.size(0), -1)
        output = self.act(self.fc1(input))
        output = self.cls(self.fc2(output))

        output = F.softmax(output, dim=1)

        return output

class Base_st_model(nn.Module):
    def __init__(self, model_name:str, input_ch:int, input_dim:int,
                conv1_kernal:int = 5, conv2_kernal:int = 5, conv1_outdim:int = 20,
                conv2_outdim:int = 20, theta_row:int=2, theta_col:int=3, fc_outdim:int=20,
                fc1_outdim:int = 32, fc2_outdim:int = 32, fc3_outdim:int = 32
                ):

        super().__init__()
        self.model_name = model_name

        self.input_ch = input_ch
        self.input_dim= input_dim
        self.conv1_kernal = conv1_kernal
        self.conv2_kernal = conv2_kernal
        self.conv1_outdim = conv1_outdim
        self.conv2_outdim = conv2_outdim 

        self.conv_out_dim = (((self.input_dim - self.conv1_kernal)+1)//2 - 
                            self.conv2_kernal)+1
        self.theta_row = theta_row
        self.theta_col = theta_col
        self.fc_outdim    = fc_outdim

        self.fc1_outdim = fc1_outdim
        self.fc2_outdim = fc2_outdim
        self.fc3_outdim = fc3_outdim

        assert model_name in ['ST-CNN', 'ST-FCN'], "model_name must be either ST-CNN or ST-FCN"
        # For ST-CNN
        if model_name == 'ST-CNN':
            self.con_loc = nn.Sequential(
                nn.Conv2d(self.input_ch, self.conv1_outdimi, self.conv1_kernal),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(self.conv1_outdimi, self.conv2_outdimi, self.conv2_kernal),
                nn.ReLU()
                )
            self.fc_loc = nn.Linear(self.conv_out_dim, self.fc_outdim)
        
        # For ST-FCN
        else:
            self.fc_loc = nn.Sequential(
                nn.Linear(self.input_ch*self.input_dim**2, self.fc1_outdim),
                nn.ReLU(),
                nn.Linear(self.fc1_outdim, self.fc2_outdim),
                nn.ReLU(),
                nn.Linear(self.fc2_outdim, self.fc3_outdim)
                )

    def forward(self, input):
        if self.model_name == 'ST-CNN':
            output = self.conv_loc(input)
            output = output.view(output.size(0), -1)
            theta = self.fc_loc(output)
            theta = theta.view(-1, self.theta_row , self.theta_col)
            
            # grid generator
            grid = F.affine_grid(theta, input.size())
            grid_sample = F.grid_sample(input, grid)
            
            return grid_sample

        else:
            input = input.view(input(0),-1)
            theta = self.fc_loc(input)
            theta = theta.view(-1, self.theta_row , self.theta_col)
            
            # grid generator
            grid = F.affine_grid(theta, input.size())
            grid_sample = F.grid_sample(input, grid)

            return grid_sample


if __name__ == '__main__()':
    import torch

    cnn = Base_cnn_model()
    rand_img = torch.randn(10,1,28,28)
    out = cnn(rand_img)
    print(out.size())
