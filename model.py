import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Base_cnn_model, Base_fcn_model, Base_st_model


class St_nn_distort(Base_cnn_model, Base_fcn_model, Base_st_model):
    def __init__(self, base_nn_model, base_st_model, base_nn_name):
        super().__init__()
        self.base_st_model = base_st_model
        self.base_nn_model = base_nn_model
        self.base_nn_name = base_nn_name

    def forward(self, input):
        output = self.base_st_model(input)
        output = self.base_nn_model(output)

        return output