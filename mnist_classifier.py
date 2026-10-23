import torch
import torch.nn as nn
from torch.functional import F
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class MnistClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lin1 = nn.Linear(64 * 7 * 7, 512)
        self.lin2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        return x if self.training else self.softmax(x)


def train():
    model = MnistClassificationModel()
    import torch.optim as optim

    num_epochs = 1000
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=10 ** -3)

    trainds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    testds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testds, batch_size=64, shuffle=False)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            output = model(images.to(device))
            loss = criterion(output, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                output = model(images)
                correct += (output.argmax(1) == labels).sum().item()
        print(f"Epoch {epoch + 1}, Accuracy: {correct / len(testds) * 100:.2f}%")


