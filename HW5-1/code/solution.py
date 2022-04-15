import argparse
from ast import arg
import torch.nn as nn
import torch.nn.functional as F
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='drop rate')
    args, _ = parser.parse_known_args()
    return args


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        ### YOUR CODE HERE
        self.conv1=nn.Conv2d(3,6,5)
        self.bn1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,16,5)
        self.bn2=nn.BatchNorm2d(16)
        self.pool=nn.MaxPool2d(2)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.drp2=nn.Dropout(args.drop)
        self.fc3=nn.Linear(84,10)
        self.rel=nn.ReLU()
        self.flt=nn.Flatten(end_dim=-1)

        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x=self.pool(self.rel(self.conv1(x)))
        x=self.bn1(x)
        x=self.pool(self.rel(self.conv2(x)))
        x=self.bn2(x)
        x=self.flt(x)
        x=self.rel(self.fc1(x))
        x=self.drp2(x)
        x=self.rel(self.fc2(x))
        x=self.drp2(x)
        x=self.fc3(x)
        return x


        ### END YOUR CODE
