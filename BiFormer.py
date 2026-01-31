
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torchvision import ops

class DilatedDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv2d_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding,
                                       dilation=dilation)
        self.deformconv2d = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                                             bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv2d_offset.weight)
        torch.nn.init.zeros_(self.conv2d_offset.bias)

    def forward(self, x):
        offset = self.conv2d_offset(x)
        return self.deformconv2d(x, offset)

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout, dilation=1):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = DilatedDeformConv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = DilatedDeformConv2d(interChannels, growthRate, kernel_size=3, padding=dilation, dilation=dilation,
                                         bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout, dilation=1):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = DilatedDeformConv2d(nChannels, growthRate, kernel_size=3, padding=dilation, dilation=dilation,
                                         bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout, dilation=1):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = DilatedDeformConv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out

class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate

        self.conv1 = DilatedDeformConv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout, dilation=1)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout, dilation=1)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout, dilation=1)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout, dilation=1)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout, dilation=1)
        nChannels += nDenseBlocks * growthRate

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout, dilation=1):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout, dilation=dilation))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout, dilation=dilation))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out
def get_interested_layer_output(model, x, layer_name):
    # 获取指定层的输出
    layer_names = [name for name, _ in model.named_children()]
    layer_index = layer_names.index(layer_name)
    intermediate_model = nn.Sequential(*list(model.children())[:layer_index + 1])
    return intermediate_model(x)

def generate_feature_map_heatmap(model, image, layer_name, channel_index):
    model.eval()

    # 转换图像为模型输入的格式
    # 你需要根据你的模型和数据预处理的方式来进行相应的操作

    # 转换为PyTorch的tensor
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

    # 获取指定层的输出
    intermediate_output = get_interested_layer_output(model, image_tensor, layer_name)

    # 提取指定通道的特征图
    feature_map = intermediate_output[0, channel_index].detach().numpy()

    # 将特征图调整为与原始图像相同的大小
    feature_map = cv2.resize(feature_map, (image.shape[1], image.shape[0]))

    # 将特征图映射为颜色
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-10)
    feature_map = (feature_map * 255).astype(np.uint8)

    # 使用特征图叠加在原始图像上
    feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.6, feature_map, 0.4, 0.)

    return result

def main():
    # 替换为你的图像路径
    image_path = "18_em_1.jpg"
    image = cv2.imread(image_path)

    # 假设你有一个已经训练好的DenseNet模型
    params = {
        'densenet': {'growthRate': 32, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': False},
        'encoder': {'input_channel': 3}
    }
    your_model = DenseNet(params)
    model_weights_path = '/liushuai2/zyh2/CB/checkpoints/5880'  # 确保这是正确的路径
    model_state_dict = torch.load(model_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    your_model.load_state_dict(model_state_dict)

    # 指定你感兴趣的层的名称，例如 'dense2'
    interested_layer_name = 'dense2'

    # 指定你感兴趣的特征图通道的索引
    channel_index = 0  # 替换为你感兴趣的通道的索引

    # 生成特征图热力图
    feature_map_heatmap_result = generate_feature_map_heatmap(your_model, image, interested_layer_name, channel_index)

    # 可视化生成的特征图热力图
    plt.imshow(feature_map_heatmap_result)
    plt.title("Feature Map Heatmap")
    plt.show()




