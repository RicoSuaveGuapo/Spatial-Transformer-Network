import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import numpy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base_model import BaseStn, BaseCnnModel
from model import StModel
from datasets import DistortedMNIST

class CAM:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.probs = None
        self.size = (model.base_nn_model.input_length, model.base_nn_model.input_length)

        self.model.eval()

        weight, _ = list(model.base_nn_model.cls.parameters())

        weight = weight.detach().numpy()
        self.weight = weight
        del weight, _

    def _forward(self, img):
        with torch.no_grad():
            features = self.model.base_nn_model.features(img) # (B, c, h', w')
            logits   = self.model.forward(img) # (B, nclass)
            probs = F.softmax(logits, dim=-1)

        self.features = features.numpy().squeeze()
        self.probs = probs.numpy().squeeze()

    def get_class_idx(self, i):
        class_idx = self.probs.argsort()
        class_idx = class_idx[-i]

        return class_idx
    
    def idx2label(self, i):
        label = [l for l in range(10)]
        class_idx = self.probs.argsort()
        class_idx = class_idx[-i]
        label = label[class_idx]
        
        return label

    def gen_heatmap(self, class_idx):
        weight_cls_i = self.weight[class_idx, :].reshape(-1,1,1) # (c, 1, 1)

        heatmap = np.sum(weight_cls_i * self.features , axis=0)
        heatmap = (heatmap - np.min(heatmap) )/(np.max(heatmap)- np.min(heatmap))
        heatmap = np.uint8(heatmap * 255)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(self.size, Image.ANTIALIAS)
        heatmap = np.array(heatmap)

        return heatmap

    def plot_heatmap(self, img, top):
        self._forward(img)
        img_numpy = img.numpy().squeeze()
        cols = top + 1 


        plt.figure(figsize= (4 * cols, 4))

        for i in range(cols):
            if i == 0:
                label = self.idx2label(i+1)

                plt.subplot(1, cols, i+1)
                plt.imshow(img_numpy, alpha=1, cmap='gray')
                plt.title(f'Original image: label = {label}')
                plt.axis('off')
            else:
                class_idx = self.get_class_idx(i)
                label = self.idx2label(i)
                probs = self.probs[class_idx]
                heatmap = self.gen_heatmap(class_idx)

                plt.subplot(1, cols, i+1)
                plt.imshow(img_numpy, alpha=1, cmap='gray')
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title(f'{label}: %.3f' % probs)
                plt.axis('off')

        plt.show()



if __name__ == '__main__':
    num = 17
    path = '/home/jarvis1121/AI/Rico_Repo/Spatial-Transformer-Network/model_save'+f'/{num}_DistortedMNIST_R_ST-CNN.pth'
    model_st = BaseStn('ST-CNN', 1, 28)
    model_cnn = BaseCnnModel(28, gap=True)
    model = StModel(model_st, model_cnn)
    model.load_state_dict(torch.load(path))
    model.eval()
    cam = CAM(model)
    

    test_img_set = DistortedMNIST(mode='test', transform_type='R', seed=42)
    test_img_loader =  DataLoader(test_img_set, batch_size=1, shuffle=True)
    for img, label in test_img_loader:
        cam.plot_heatmap(img, top=3)
        break




    # print(cam.gen_heatmap().shape)
    # plt.imshow(cam.gen_heatmap)
    # plt.show()


    


