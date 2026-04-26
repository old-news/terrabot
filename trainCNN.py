from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import sys
import time
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
import math
from PIL import Image
if __name__ == '__main__': print("Done importing")


class defaultCNN(nn.Module):
    def __init__(self):
        super(defaultCNN, self).__init__()
        self.epochs = 0
        self.epochMessages = []
        self.datamean = None
        self.datastd = 0
        self.trainingTransform = transforms.Compose([
            transforms.Resize(self.inputReshape),  # Not necessary, but just in case
            transforms.ToTensor(),
        ])

    def forward(self, x):
        y = self.features(x)
        return self.classifier(y)

    def calculateNormalizationData(self):
        print('Calculating normalization data...')
        self.datamean = 0
        self.datastd = 0
        dataset = datasets.ImageFolder(self.trainpath, transform=self.trainingTransform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            self.datamean += images.mean(2).sum(0)
            self.datastd += images.std(2).sum(0)

        self.datamean /= len(loader.dataset)
        self.datastd /= len(loader.dataset)
        self.transform = transforms.Compose([
            #transforms.Resize(self.inputReshape),  # Not necessary, but just in case
            #transforms.ColorJitter(brightness=0.4, contrast=0.4),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=self.datamean, std=self.datastd)
        ])

    def trainModel(self, epochs=None):
        if self.datamean is None:
            self.datamean = 0
            self.calculateNormalizationData()
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        print('Indexing training data...')
        gputransform = v2.Compose([
            v2.Resize(self.inputReshape),
            v2.ColorJitter(brightness=0.4, contrast=0.4),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            # Convert to float and scale to [0, 1] if not already done
            #v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.datamean, std=self.datastd)
        ])
        dataset = datasets.ImageFolder(self.trainpath, transform=self.transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        runningLoss = 0.0
        self.train()
        print(f"Training on {device}...")
        for epoch in range(epochs):
            runningLoss = 0.0
            start = time.perf_counter()
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                self.optimizer.zero_grad()
                images = gputransform(images)
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.detach()
            self.scheduler.step(runningLoss)
            averageLoss = runningLoss.item() / len(loader)
            self.epochMessages.append(f"Epoch {self.epochs+1:>6} | Loss: {averageLoss:.6f} | Training accuracy: {100 * (math.e ** -averageLoss):.3f}% | Elapsed: {time.perf_counter() - start:.2f}")
            print(self.epochMessages[-1])
            self.epochs+=1
            if self.epochs % 10 == 0: self.save()
        self.eval()

    def save(self):
        torch.save({
            'epochs': self.epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'datamean': self.datamean,
            'datastd': self.datastd,
            'epochMessages': self.epochMessages,
            'transform': self.transform
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
            self.epochMessages = checkpoint['epochMessages']
            self.transform = checkpoint['transform']
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
            nn.MaxPool2d(4, 4),
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
        if blocktype is None: blocktype = 'tile'
        self.savepath = f'./nn/{blocktype}.cnn'
        self.trainpath = f'./training/{blocktype}'
        self.inputShape = (48, 48)
        self.inputReshape = (16, 16)
        super(blockCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        headSize = 32 * int(self.inputReshape[0] / 2) * int(self.inputReshape[1] / 2)
        self.classifier = nn.Linear(headSize, len(os.listdir(self.trainpath)))
        numCategories = len(os.listdir(self.trainpath))
        self.categoryHead = nn.Linear(headSize, numCategories)
        ids = set()
        for category in os.listdir(self.trainpath):
            catPath = os.path.join(self.trainpath, category)
            for ID in os.listdir(catPath):
                ids.add(os.path.join(catPath, ID))
        self.idHead = nn.Linear(headSize, len(ids))
        #self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=numCategories, batch_first=True)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def otherforward(self, x):
        x = self.features(x)
        batchSize, channels, h, w = x.shape
        x = x.view(batchSize, channels, h*w)
        x = x.permute(0, 2, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(0, 2, 1)
        x = x.view(batchSize, channels, h, w)
        return x
        y = self.features(x)
        catPred = self.categoryHead(y)
        idPred = self.idHead(y)
        return catPred, idPred

    def modelTrain(self, epochs=None):
        if self.datamean is None:
            self.datamean = 0
            self.calculateNormalizationData()
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        print('Indexing training data...')
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
                predicted = self(images)
                loss = criterion(predicted, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
            self.scheduler.step(runningLoss)
            self.epochMessages.append(f"Epoch {self.epochs+1:>6} | Loss: {runningLoss / len(loader):.6f} | Elapsed: {time.perf_counter() - start:.6f}")
            print(self.epochMessages[-1])
            self.epochs+=1
            if self.epochs % 10 == 0: self.save()
        self.eval()


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
        self.ids = sorted(list(unique_ids))
        self.id2idx = {ID: i for i, ID in enumerate(self.ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagePath, categoryIndex, ID = self.samples[idx]
        image = Image.open(imagePath)
        #image = cv2.imread(imagePath)
        if self.transform: image = self.transform(image)
        id_idx = self.id2idx[ID]
        return image, categoryIndex, id_idx


class multiHeadModel(blockCNN):
    def __init__(self):
        self.savepath = './nn/mhm.cnn'
        super().__init__()
        numCategories = len(os.listdir('./training/tile'))
        ids = set()
        for category in os.listdir('./training/tile'):
            categoryPath = os.path.join('./training/tile', category)
            for ID in os.listdir(categoryPath):
                ids.add(ID)
        numIDs = len(ids)
        self.categoryHead = nn.Linear(512, numCategories)
        self.idHead = nn.Linear(512, numIDs)
        self.trainingTransform = transforms.Compose([
            transforms.Resize(self.inputReshape),  # Not necessary, but just in case
            transforms.ToTensor(),
        ])

    def forward(self, x):
        features = self.features(x)
        category = self.categoryHead(features)
        ID = self.idHead(features)
        return category, ID

    def calculateNormalizationData(self):
        print('Calculating normalization data...')
        dataset = multiHeadDataset('./training/tile', self.trainingTransform)
        dataLoader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        for images, cat, ID in dataLoader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            self.datamean += images.mean(2).sum(0)
            self.datastd += images.std(2).sum(0)

        self.datamean /= len(dataset)
        self.datastd /= len(dataset)
        self.transform = transforms.Compose([
            transforms.Resize(self.inputReshape),  # Not necessary, but just in case
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.datamean, std=self.datastd)
        ])

    def trainModel(self, epochs=None):
        if self.datamean is None:
            self.datamean = 0
            self.calculateNormalizationData()
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        dataset = multiHeadDataset('./trainig/tile', self.trainingTransform)
        dataLoader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            start = time.perf_counter()
            runningLoss = 0
            catlabels = []
            idlabels = []
            catpred = []
            idpred = []
            for images, category, ID in dataLoader:
                images, category, ID = images.to(device), category.to(device), ID.to(device)
                catlabels.append(category)
                idlabels.append(ID)
                self.optimizer.zero_grad()
                categoryPredicted, idPredicted = self(images)
                catpred.append(categoryPredicted)
                idpred.append(idPredicted)
                categoryLoss = criterion(categoryPredicted, category)
                idLoss = criterion(idPredicted, ID)
                total_loss = categoryLoss + idLoss
                total_loss.backward()
                self.optimizer.step()
                runningLoss += total_loss
            confusionMatrix = confusion_matrix(catlabels, catpred)
            plt.figure(figsize=(10, 10))
            sns.heatmap(confusionMatrix, annot=False, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            self.scheduler.step(runningLoss)
            print(f"Epoch {self.epochs+1:>6} | Loss: {runningLoss / len(dataLoader):.6f} | Elapsed: {time.perf_counter() - start:.6f}")
            self.epochs+=1
            if self.epochs % 10 == 0: self.save()
        self.eval()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        net = blockCNN(blocktype='tile')
        #net.calculateNormalizationData()
        net.load()
        # Below is for Colab training
        #net.savepath = './content/drive/MyDrive/terrabot/nn/tile.cnn'
        net = torch.compile(net)
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
