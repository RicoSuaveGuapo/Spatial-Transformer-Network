import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_cnn_model(nn.Module):
    def __init__(self, input_ch:int=1, input_length:int, conv1_out_fea:int=32, conv2_out_fea:int=64):
        super().__init__()
        
        # the number of filters shold be around 32~64
        assert 32<= conv1_out_fea <= 64, 'number of filters in conv1 must in the range of 32~64'
        assert 32<= conv2_out_fea <= 64, 'number of filters in conv1 must in the range of 32~64'
        
        self.conv1_out_fea = conv1_out_fea
        self.conv2_out_fea = conv2_out_fea

        # the number of learnable parameter should be around 400,000
        assert 399500 <= self.num_para() <= 400500, 'number of parameters should around 400k'

        self.conv1 = nn.Conv2d(input_ch, self.conv1_out_fea, 9) # (self.conv1_out_fea, input_length-9+1, same)
        self.pool1 = nn.MaxPool2d(2) # (self.conv1_out_fea, (input_length-9+1)//2, same)
        self.act   = nn.ReLU()
        self.conv2 = nn.Conv2d(self.conv1_out_fea, self.conv2_out_fea, 7) # (self.conv2_out_fea, (input_length-9+1)//2-7+1, same)
        self.pool2 = nn.MaxPool2d(2) # (self.conv2_out_fea, ((input_length-9+1)//2-7+1)//2, same)
        self.cls = nn.Linear(self.conv2_out_fea * (((input_length-9+1)//2-7+1)//2)**2, 10)

    
    def forward(self, input):
        output = self.act(self.pool1(self.conv1(input)))
        output = self.pool2(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.cls(output)

        output = F.softmax(output, dim=1)

        return output

    def num_para(self):
        cnn_para = (9*9*self.conv1_out_fea) + (7*7*self.conv2_out_fea) + (10)
        return int(cnn_para)


class Base_fcn_model(nn.Module):
    def __init__(self, input_ch:int=1, input_length:int, fc1_unit:int=128, fc2_unit:int=256):
        super().__init__()
        # the number of unit should be around 128~256
        assert 128 <= fc1_unit <= 256, 'number of unit in fc1 should around 128~256'
        assert 128 <= fc2_unit <= 256, 'number of unit in fc2 should around 128~256'

        self.fc1_unit = fc1_unit
        self.fc2_unit = fc2_unit
        
        assert 399500 <= self.num_para() <= 400500, 'number of parameters should around 400k'

        self.fc1 = nn.Linear(input_ch*input_length*input_length, self.fc1_unit)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_unit, self.fc2_unit)
        self.cls = nn.Linear(self.fc2_unit, 10)

    def forward(self, input):

        input = input.view(input.size(0), -1)
        output = self.act(self.fc1(input))
        output = self.cls(self.fc2(output))

        output = F.softmax(output, dim=1)
        return output

    def num_para(self):
        fcn_para = self.fc1_unit + self.fc2_unit
        return int(fcn_para)


class Base_stn(nn.Module):
    def __init__(self, model_name:str, input_ch:int, input_dim:int,
                conv1_kernal:int = 5, conv2_kernal:int = 5, conv1_outdim:int = 20,
                conv2_outdim:int = 20, theta_row:int=2, theta_col:int=3, fc_outdim:int=20,
                fc1_outdim:int = 32, fc2_outdim:int = 32, fc3_outdim:int = 32, trans_type:str = 'Aff'
                ):
        assert model_name in ['ST-CNN', 'ST-FCN'], "model name must be either ST-CNN or ST-FCN"
        assert trans_type in ['Aff','Proj','TPS'], 'transformation_type must be one of Aff, Proj, TPS'
        
        super().__init__()
        self.model_name = model_name
        self.trans_type = trans_type

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
        self.fc_outdim = fc_outdim

        self.fc1_outdim = fc1_outdim
        self.fc2_outdim = fc2_outdim
        self.fc3_outdim = fc3_outdim

        # --localisation networks --
        # For ST-CNN
        if model_name == 'ST-CNN':
            self.con_loc = nn.Sequential(
                nn.Conv2d(self.input_ch, self.conv1_outdim, self.conv1_kernal),     # (20, 24, 24)
                nn.MaxPool2d(2),                                                    # (20, 12, 12)
                nn.ReLU(),
                nn.Conv2d(self.conv1_outdimi, self.conv2_outdim, self.conv2_kernal), # (20, 8, 8)
                nn.ReLU()
                )
            self.fc_loc = nn.Linear(self.conv_out_dim, self.fc_outdim)               # (20)
        
        # For ST-FCN
        else:
            self.fc_loc = nn.Sequential(
                nn.Linear(self.input_ch*self.input_dim**2, self.fc1_outdim),         # (32)
                nn.ReLU(),
                nn.Linear(self.fc1_outdim, self.fc2_outdim),                         # (32)
                nn.ReLU(),
                nn.Linear(self.fc2_outdim, self.fc3_outdim)                          # (32)
                )


    def forward(self, input):
        if self.model_name == 'ST-CNN':
            output = self.conv_loc(input)
            output = output.view(output.size(0), -1)
            theta = self.fc_loc(output)
            theta = theta.view(-1, self.theta_row , self.theta_col)
            
            # grid generator
            if self.trans_type == 'Aff':
                grid = F.affine_grid(theta, input.size())
                grid_sample = F.grid_sample(input, grid)
            
                return grid_sample

            elif self.trans_type == 'Proj':
                #TODO
                pass
            else:
                #TODO
                pass

        else:
            input = input.view(input(0),-1)
            theta = self.fc_loc(input)
            theta = theta.view(-1, self.theta_row , self.theta_col)
            
            # grid generator
            grid = F.affine_grid(theta, input.size())
            grid_sample = F.grid_sample(input, grid)

            return grid_sample

    def num_para(self):
        #TODO
        pass

if __name__ == '__main__()':
    import torch

    cnn = Base_cnn_model()
    rand_img = torch.randn(10,1,28,28)
    out = cnn(rand_img)
    print(out.size())
