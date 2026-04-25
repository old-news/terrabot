import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os, sys
import time
if __name__ == '__main__': print("Done importing")

class defaultCNN(nn.Module):
    def __init__(self):
        super(defaultCNN, self).__init__()
        self.epochs = 0
        self.datamean = 0
        self.datastd = 0
        transform = transforms.Compose([
            transforms.Resize(self.inputReshape), # Not necessary, but just in case
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(self.trainpath, transform=transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        self.datamean /= len(loader.dataset)
        self.datastd /= len(loader.dataset)
        self.transform = transforms.Compose([
            transforms.Resize(self.inputReshape), # Not necessary, but just in case
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.datamean, std=self.datastd)
        ])

    def forward(self, x):
        y = self.features(x)
        return self.classifier(y)
    
    def trainModel(self, epochs=None):
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        dataset = datasets.ImageFolder(self.trainpath, transform=self.transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        runningLoss = 0.0
        self.train()
        print("Training...")
        for epoch in range(epochs):
            runningLoss = 0.0
            start = time.perf_counter()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
            self.scheduler.step(runningLoss)
            print(f"Epoch {self.epochs+1:>6} | Loss: {runningLoss / len(loader):.6f} | Elapsed: {time.perf_counter() - start:.6f}")
            self.epochs+=1
            if self.epochs % 10 == 0: self.save()
        self.eval()

    def save(self):
        torch.save({
            'epochs': self.epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            'datamean': self.datamean,
            'datastd': self.datastd
        }, self.savepath)
        print("model saved")

    def load(self):
        try:
            checkpoint = torch.load(self.savepath, weights_only=False)
            self.epochs = checkpoint['epochs']
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.datamean = checkpoint['datamean']
            self.datastd = checkpoint['datastd']
        except:
            pass
    
    def run(self, inputImage):
        # The model is trained in batches, which add an extra dimension
        # Unsqueeze(0) adds a dummy dimension for the dimensions to match when the model runs
        transformedImage = self.transform(inputImage).unsqeeze(0)
        self.eval()
        with torch.no_grad():
            return self(transformedImage)



class offsetCNN(defaultCNN):
    def __init__(self):
        self.savepath = './nn/offset.cnn'
        self.trainpath = './training/offset'
        self.inputShape = (200, 200)
        self.inputReshape = (200, 200)
        super(offsetCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            #nn.Conv2d(16, 32, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.classifier = nn.Linear(16 * int(self.inputReshape[0] / 4) * int(self.inputReshape / 4), len(os.listdir(self.trainpath)))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

class blockCNN(defaultCNN):
    def __init__(self, blocktype=None):
        if blocktype is None: self.blocktype = blocktype
        self.savepath = f'./nn/{blocktype}.cnn'
        self.trainpath = f'./training/{blocktype}'
        self.inputShape = (48, 48)
        self.inputReshape = (16, 16)
        super(blockCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.classifier = nn.Linear(32 * int(self.inputReshape[0] / 2) * int(self.inputReshape[1] / 2), len(os.listdir(self.trainpath)))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

class multiHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.categories = sorted([directory for directory in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, directory))])
        self.category2idx = {category: i for i, category in enumerate(self.categories)}
        unique_ids = set()
        
        for category in self.categories:
            categoryPath = os.path.join(root_dir, category)
            ids = sorted([directory for directory in os.listdir(categoryPath) if os.path.isdir(os.path.join(categoryPath, directory))])
            for ID in ids:
                unique_ids.add(ID)
                path = os.path.join(categoryPath, ID)
                for imgName in os.listdir(path):
                    self.samples.append((os.path.join(path, imgName), self.category2idx[category], ID))
        unique_ids = sorted(list(unique_ids))
        self.id2idx = {ID: i for i, ID in enumerate(unique_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagePath, categoryIndex, ID = self.samples[idx]
        image = cv2.imread(imagePath)
        if self.transform: image = self.transform(image)
        id_idx = self.id2idx[ID]
        return image, categoryIndex, id_idx

class multiHeadModel(nn.Module):
    def __init__(self, numCategories, numIDs):
        super().__init__()
        self.categoryHead = nn.Linear(512, numCategories)
        self.idHead = nn.Linear(512, numIDs)

    def forward(self, x):
        features = self.features(x)
        category = self.categoryHead(features)
        ID = self.idHead(features)
        return category, ID

    def trainModel(self):
        categoryPredicted, idPredicted = self(images)
        categoryLoss = criterion(categoryPredicted, categoryTargets)
        idLoss = criterion(idPredicted, idTargets)
        total_loss = categoryLoss + idLoss
        total_loss.backward()
        optimizer.step()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        net = blockCNN(blocktype='tile')
        net.load()
        net.trainModel()
        exit(0)
    if sys.argv[1] in ['tile', 'air', 'wall', 'liquid']:
        net = blockCNN(blocktype=sys.argv[1])
        net.load()
        net.trainModel()
    elif sys.argv[1] == 'offset':
        net = offsetCNN()
        net.load()
        net.trainModel()
