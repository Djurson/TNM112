import torch.nn as nn
import residualblock

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        # Initial Prep: Keep 32x32 resolution
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.block1 = residualblock.ResidualBlock(32, 32, stride=1)
        self.block2 = residualblock.ResidualBlock(32, 64, stride=2)
        self.block3 = residualblock.ResidualBlock(64, 128, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.dense_block = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2), 
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.prep(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        return x