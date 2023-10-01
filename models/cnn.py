import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = F.relu(x)
        out_list.append(out)
        out = self.conv2(x)
        out_list.append(out)
        out = F.relu(x)
        out_list.append(out)
        out = F.max_pool2d(x, 2)
        out_list.append(out)
        out = self.dropout1(x)
        out_list.append(out)
        out = torch.flatten(x, 1)
        out_list.append(out)
        out = self.fc1(x)
        out_list.append(out)
        out = F.relu(x)
        out_list.append(out)
        out = self.dropout2(x)
        out_list.append(out)
        out = self.fc2(x)
        out_list.append(out)
        y = F.log_softmax(x, dim=1)
        return y, out_list

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        penultimate = F.relu(out)
        out = self.dropout2(penultimate)
        out = self.fc2(out)
        y = F.log_softmax(out, dim=1)

        return y, penultimate
