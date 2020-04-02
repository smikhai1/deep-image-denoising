import torch.nn as nn
import torch.nn.functional as F

class RED_Net_30(nn.Module):
    """
    This baseline is 30-layered residual encoder-decoder neural network
    with symmetric skip-connections between convolutional and deconvolutional
    layers with step 2, ReLU activations, filters of constant size 3x3, constant
    number of channels (128) in activations of each layer, padding = 1, stride = 1,
    no max-pooling.
    """

    def __init__(self):
        super(RED_Net_30, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_10 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_11 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_12 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_13 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_14 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_15 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_6 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_7 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_8 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_9 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_10 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_11 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_12 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_13 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_14 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv_15 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, X):
        X = self.conv_1(X)
        X_2 = self.conv_2(X)

        X = self.conv_3(X_2)
        X_4 = self.conv_4(X)

        X = self.conv_5(X_4)
        X_6 = self.conv_6(X)

        X = self.conv_7(X_6)
        X_8 = self.conv_8(X)

        X = self.conv_9(X_8)
        X_10 = self.conv_10(X)

        X = self.conv_11(X_10)
        X_12 = self.conv_12(X)

        X = self.conv_13(X_12)
        X_14 = self.conv_14(X)

        X = self.conv_15(X_14)
        X = self.deconv_1(X)

        X = self.deconv_2(F.relu(X + X_14))
        X = self.deconv_3(X)

        X = self.deconv_4(F.relu(X + X_12))
        X = self.deconv_5(X)

        X = self.deconv_6(F.relu(X + X_10))
        X = self.deconv_7(X)

        X = self.deconv_8(F.relu(X + X_8))
        X = self.deconv_9(X)

        X = self.deconv_10(F.relu(X + X_6))
        X = self.deconv_11(X)

        X = self.deconv_12(F.relu(X + X_4))
        X = self.deconv_13(X)

        X = self.deconv_14(F.relu(X + X_2))
        X = self.deconv_15(X)

        return X

class RED_Net_20(nn.Module):
    """
    This baseline is 20-layered residual encoder-decoder neural network
    with symmetric skip-connections between convolutional and deconvolutional
    layers with step 2, ReLU activations, filters of constant size 3x3, constant
    number of channels (128) in activations of each layer, padding = 1, stride = 1,
    no max-pooling.
    """

    def __init__(self):
        super(RED_Net_20, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_6 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_9 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv_10 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.conv_1(X)
        X = self.conv_2(X)

        X_3 = self.conv_3(X)
        X = self.conv_4(X_3)

        X_5 = self.conv_5(X)
        X = self.conv_6(X_5)

        X_7 = self.conv_7(X)

        X = self.conv_8(X_7)
        X_9 = self.conv_9(X)

        X = self.conv_10(X_9)
        X = self.deconv_1(X)

        X = self.deconv_2(F.relu(X + X_9))
        X = self.deconv_3(X)

        X = self.deconv_4(F.relu(X + X_7))
        X = self.deconv_5(X)

        X = self.deconv_6(F.relu(X + X_5))
        X = self.deconv_7(X)

        X = self.deconv_8(F.relu(X + X_3))
        X = self.deconv_9(X)

        X = self.deconv_10(X)

        return X