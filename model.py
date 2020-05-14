import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_cnn_model(nn.Module):
    # TODO, based on model!
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

class

class Base_st_model(nn.Module):
    def __init__(self, input_ch:int, input_dim:int):
        super().__init__()
        self.input_ch = input_ch
        self.input_dim= input_dim
        self.conv_out_dim = ((((self.input_dim-7)+1)//2 -5) +1)//2

        # conv layers
        # TODO modify here for more flexiable
        self.localisation = nn.Sequential(
            nn.Conv(self.input_ch, 8, kernel_size=7),
            nn.Maxpool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv(8, 10, kernel_size=5),
            nn.Maxpool2d(2, stride=2),
            nn.ReLU(True)
            )

        # fc layers
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.conv_out_dim ** 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )

    def forward(self, input):
        output = self.localisation(input)
        output = output.view(output.size(0), -1)
        theta = self.fc_loc(output)
        theta = theta.view(-1, 2, 3) # TODO not only (2,3)
        
        # grid generator
        grid = F.affine_grid(theta, input.size())
        grid_sample = F.grid_sample(input, grid)
        
        return grid_sample

class 


if __name__ == '__main__()':
    import torch

    cnn = base_cnn_model()
    rand_img = torch.randn(10,1,28,28)
    out = cnn(rand_img)
    print(out.size())
