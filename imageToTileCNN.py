import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os, sys
print("Done importing")

class image2TileCNN(nn.Module):
    def __init__(self):
        super(image2TileCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 8 * 8, len(os.listdir('./training/tile')))
        self.epochs = 0
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def forward(self, x):
        y = self.features(x)
        return self.classifier(y)

    def trainModel(self, epochs=None):
        print("Training...")
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        transform = transforms.Compose([
            transforms.Resize((16, 16)), # Must match image size (16x16)
            transforms.ToTensor(),
        ])

        dataset = datasets.ImageFolder('./training/tile', transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        runningLoss = 0.0
        self.train()
        for epoch in range(epochs):
            runningLoss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
            self.scheduler.step(runningLoss)
            print(f"Epoch {self.epochs+1} | Loss: {runningLoss / len(loader):.7f}")
            self.epochs+=1
            if self.epochs % 10 == 0:
                self.save()
        self.eval()

    def save(self):
        torch.save({
            'epochs': self.epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, 'img2Tile.cnn')

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.epochs = checkpoint['epochs']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if __name__ == '__main__':
    net = image2TileCNN()
    try:
        net.load('img2Tile.cnn')
    except:
        pass
    net.trainModel()
