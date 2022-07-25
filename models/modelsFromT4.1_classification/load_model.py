import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f


PATH = r'C:\Users\bachtses\Documents\WebProjects\Chest X-Ray Medical Report Web App (DEVELOPMENT)\models\modelsFromT4.1_classification\acr_class_weights_29_4.pth'


class ConvNeuralNet(nn.Module):


    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_1 = nn.Dropout2d(0.5)

        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(32 * 10 * 10, num_classes)


    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.drop_1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = self.drop_2(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)

        return out


models = ConvNeuralNet(2)
print(models)

# Initialize optimizer
optimizers = optim.SGD(models.parameters(), lr=0.001, momentum=0.9)
models.load_state_dict(torch.load(PATH))

print("Models state_dict:")
for param_tensor in models.state_dict():
    print(param_tensor, "\t", models.state_dict()[param_tensor].size())


print("Optimizers state_dict:")
for var_name in optimizers.state_dict():
    print(var_name, "\t", optimizers.state_dict()[var_name])
torch.save(models.state_dict(), PATH)

models.eval()

