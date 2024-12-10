import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, num_filters, image_size):
        super(CNN, self).__init__()

        self.conv=torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=(image_size//16, 2), stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=(4, 4), stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=num_filters*2),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        hidden_layer_size=10

        self.mlp=nn.Sequential(
            nn.Linear(in_features=1, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc=torch.nn.Sequential(
            nn.Linear(in_features=175114, out_features=150),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=150, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, image, sf_vol_frac):
        conv_output=self.conv(image)
        mlp_output=self.mlp(sf_vol_frac.unsqueeze(1))
        return self.fc(torch.cat([conv_output, mlp_output], dim=1))
