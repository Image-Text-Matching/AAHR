import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bn=True, ln=False):
        super().__init__()

        self.output_dim = output_dim
        self.num_layers = num_layers
        # 创建了一个包含隐藏层维度的列表
        h = [hidden_dim] * (num_layers - 1)
        self.bn = bn
        self.ln = ln

        # 构建MLP的各个隐藏层和输出层
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if self.bn:
            # 创建 BN 层
            self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):

        if len(x.size()) == 3 and self.bn:
            B, N, D = x.size()
            # The data size has been changed for the compatibility of BatchNorm Layer,
            # 为了兼容BatchNorm层，更改了数据大小，
            # The original data shape is B*N*D, 
            # while the bn layer needs the data whose shape is B*D
            x = x.reshape(B*N, D)
        else:
            N = 0

        if self.bn:
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                # 应用线性变换后再通过激活函数ReLU进行非线性变换，最后一个线性层不需要ReLU激活函数
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        else:
            for i, layer in enumerate(self.layers):
                # 应用线性变换后再通过激活函数ReLU进行非线性变换，最后一个线性层不需要ReLU激活函数
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)        
        
        if N > 0:
            x = x.view(B, N, self.output_dim)

        return x


class FC_MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bn=True):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, bn)
    
    def forward(self, x):

        # 残差连接
        x = self.fc(x) + self.mlp(x)
        return x


if __name__ == '__main__':
    
    pass
