import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

class AgeAlexNet(nn.Module):
    def __init__(self, pretrained=False, modelpath=None):
        super(AgeAlexNet, self).__init__()
        
        # Load the pretrained AlexNet model
        self.alexnet = models.alexnet(pretrained=pretrained)

        # Modify the classifier to predict age
        if pretrained:
            self._load_pretrained_params(modelpath)

        # Replace the final layer with a new one for age classification
        self.alexnet.classifier[6] = nn.Linear(4096, 5)  # 5 classes for age
        
    def forward(self, x):
        # Forward pass through the AlexNet model
        return self.alexnet(x)

    def _load_pretrained_params(self, path):
        # Load custom pretrained weights from file if provided
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

class AgeClassify:
    def __init__(self, pretrained=False, modelpath=None):
        # Define model (use pretrained AlexNet with age classifier)
        self.model = AgeAlexNet(pretrained=pretrained, modelpath=modelpath).cuda()

        # Define optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # Define loss function
        self.criterion = nn.CrossEntropyLoss().cuda()

    def train(self, input, label):
        self.model.train()
        output = self.model(input)
        self.loss = self.criterion(output, label)

    def val(self, input):
        self.model.eval()
        with torch.no_grad():
            output = F.softmax(self.model(input), dim=1).max(1)[1]
        return output

    def save_model(self, dir, filename):
        torch.save(self.model.state_dict(), os.path.join(dir, filename))

if __name__ == "__main__":
    # Example usage
    one = torch.ones((1, 3, 224, 224)).cuda()  # Correct input size for AlexNet
    model = AgeAlexNet(pretrained=True)  # Load pretrained weights
    output = model(one)
    print(output)
