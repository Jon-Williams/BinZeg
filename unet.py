import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def DoubleConvolution(inputChannels, outputChannels):
    convolution = nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputChannels, outputChannels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )
    return convolution

class Unet(nn.Module):
    def __init__(self, inputChannels=1, outputChannels=1, featureMap=[64, 128, 256, 512, 1024]):
        super(Unet, self).__init__()
        self.downFeatureMap = nn.ModuleList()
        self.upFeatureMap = nn.ModuleList()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inDimension = inputChannels

        # Encoder operations
        for feature in featureMap[:-1]:
            # print('Convolution from ' + str(self.inDimension) + ' to ' + str(feature))
            self.downFeatureMap.append(DoubleConvolution(self.inDimension, feature))
            self.inDimension = feature

        # Decoder operations
        for feature in reversed(featureMap[:-1]):
            # print('Transpose convolution from ' + str(self.inDimension) + ' to ' + str(feature))
            self.upFeatureMap.append(nn.ConvTranspose2d(self.inDimension, feature, kernel_size=2, stride=2))
            self.upFeatureMap.append(DoubleConvolution(self.inDimension, feature))
            self.inDimension = feature

        # Middle and final operations
        self.bottleNeck = DoubleConvolution(featureMap[-2], featureMap[-1])
        self.finalConv = nn.Conv2d(featureMap[0], outputChannels, kernel_size=1)

    def forward(self, image):
        skipConnection = []

        # Encode
        for encode in self.downFeatureMap:
            image = encode(image)
            skipConnection.append(image)
            image = self.maxPool(image)

        image = self.bottleNeck(image)

        # Decode
        for i in range(0, len(self.upFeatureMap), 2):
            image = self.upFeatureMap[i](image)
            copyIndex = skipConnection[i//2]

            if image.shape != skipConnection.shape:
                x = TF.resize(img, size=skipConnection.shape[2:])

            
        return image




def main():
    x = torch.randn((1,1,161,161))
    test = Unet(1,1)
    preds = test(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    main()
