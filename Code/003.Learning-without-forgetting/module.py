from torch import nn, optim

# Source code of AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(  # [3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [384, 12, 12]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [256, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [256, 5, 5]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),    # initial as 2
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output