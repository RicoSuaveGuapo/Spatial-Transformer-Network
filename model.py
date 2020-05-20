import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseCnnModel, BaseFcnModel, BaseStn

class CnnModel(nn.Module):
    #TODO the number of filiters are b/t 32~64
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 9) #(32, 20, 20) 
        self.pool1 = nn.MaxPool2d(2)    #(32, 10, 10)
        self.act   = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7) #(64, 4, 4)
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


class FcnModel(nn.Module):
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


class StModel(nn.Module):
    def __init__(self, base_stn, base_nn_model):
        super().__init__()
        self.base_stn = base_stn
        self.base_nn_model = base_nn_model
        
    def forward(self, input):
        output = self.base_stn(input)       # (N, 28, 28)
        output = self.base_nn_model(output) # (N, 10)

        return output


if __name__ == '__main__':
    rand_img = torch.randn(11,1,28,28)

    stn = BaseStn(model_name='ST-FCN', input_ch=rand_img.size(1) , input_length=rand_img.size(2))
    base_fcn = BaseFcnModel(input_length=rand_img.size(2))

    st_fcn = StModel(base_stn = stn, base_nn_model = base_fcn)
    output = st_fcn(rand_img)
    print(output.size())
