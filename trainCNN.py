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
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
#from skimage import io
import math
import numpy as np
from PIL import Image
import functools
import random
import matplotlib.pyplot as plt
if __name__ == '__main__': print("Done importing")


class defaultCNN(nn.Module):
    def __init__(self):
        super(defaultCNN, self).__init__()
        self.epochs = 0
        self.epochMessages = []
        self.datamean = None
        self.datastd = 0
        self.accuracy = 0
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
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=self.datamean, std=self.datastd)
        ])
        self.validationDataset = None
        self.validationLoader = None
        self.device = None

    def trainModel(self, epochs=None):
        if self.datamean is None:
            self.datamean = 0
            self.calculateNormalizationData()
        if epochs is None: epochs = sys.maxsize * sys.maxsize
        print('Indexing training data...')
        gputransform = v2.Compose([
            #v2.ToTensor(),
            v2.Resize(self.inputReshape),
            v2.ColorJitter(brightness=0.4, contrast=0.4),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            # Convert to float and scale to [0, 1] if not already done
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.datamean, std=self.datastd)
        ])
        #dataset = multiHeadDataset(transform=self.transform)  # datasets.ImageFolder(self.trainpath, transform=self.transform)
        dataset = flattenDataset(self.trainpath, transform=self.trainingTransform, target_transform=gputransform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        self.train()
        print(f"Training on {self.device}...")
        for epoch in range(epochs):
            runningLoss = 0
            total = 0
            correct = 0
            start = time.perf_counter()
            for images, idlabels in loader:
                images, idlabels = images.to(self.device, non_blocking=True), idlabels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                ids = self(images)
                loss = criterion(ids, idlabels) * 5 / 5  # criterion(cats, catlabels) / 5 + criterion(ids, idlabels) * 4 / 5
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
                _, pred = torch.max(ids, 1)
                total += idlabels.size(0)
                correct += (pred == idlabels).sum().item()
            self.scheduler.step(runningLoss)
            averageLoss = runningLoss / len(loader)
            self.epochMessages.append(f"Epoch {self.epochs+1:>6} | Loss: {averageLoss:.6f} | Training accuracy: {100 * correct/total:.3f}% | Elapsed: {time.perf_counter() - start:.2f}")
            print(self.epochMessages[-1])
            self.epochs+=1
            if self.epochs % 10 == 0:
                accuracy = self.validation()
                if accuracy > self.accuracy:
                    self.accuracy = accuracy
                    self.save()
            #confusionMatrix = confusion_matrix(np.array(all_labels), np.array(all_predicted))
            #plt.figure(figsize=(10, 10))
            #sns.heatmap(confusionMatrix, annot=False, cmap="Blues")
            #plt.xlabel("Predicted")
            #plt.ylabel("Actual")
            #plt.show()
        self.eval()

    def validation(self):
        # datasets.ImageFolder(self.trainpath, transform=self.transform)
        gputransform = v2.Compose([
            v2.Resize(self.inputReshape),
            # Convert to float and scale to [0, 1] if not already done
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.datamean, std=self.datastd)
        ])
        if self.validationDataset is None:
            print("Indexing validation data...")
            self.validationDataset = flattenDataset(self.validationpath, transform=self.trainingTransform, target_transform=gputransform)
            self.validationLoader = DataLoader(self.validationDataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        self.eval()
        print("validating...")
        with torch.no_grad():
            total = 0
            correct = 0
            start = time.perf_counter()
            for images, labels in self.validationLoader:
                images, images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self(images)
                #loss = criterion(ids, idlabels) * 5 / 5  # criterion(cats, catlabels) / 5 + criterion(ids, idlabels) * 4 / 5
                #runningLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct/total
            print(f"Validation   = Loss: uncalcd. = Validate accuracy: {100 * accuracy:.3f}% = Elapsed: {time.perf_counter() - start:.2f}")
            #confusionMatrix = confusion_matrix(np.array(all_labels), np.array(all_predicted))
            #plt.figure(figsize=(10, 10))
            #sns.heatmap(confusionMatrix, annot=False, cmap="Blues")
            #plt.xlabel("Predicted")
            #plt.ylabel("Actual")
            #plt.show()
        self.train()
        return accuracy

    def save(self):
        torch.save({
            'epochs': self.epochs,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'datamean': self.datamean,
            'datastd': self.datastd,
            'epochMessages': self.epochMessages,
            'transform': self.transform,
            'features': self.features,
            'accuracy': self.accuracy
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
            self.features = checkpoint['features']
            self.accuracy = checkpoint['accuracy']
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
    def __init__(self, trainpath=None):
        self.name = 'block'
        if trainpath is not None and trainpath.replace('./', '').replace('/', '') != 'training':
            self.name = ''.join(trainpath.split('training')[1:])[1:]
        if trainpath is None: trainpath = './training'
        self.trainpath = trainpath
        self.validationpath = trainpath.replace('training', 'validation')
        self.savepath = f'./nn/{self.name}'
        self.inputShape = (16, 16)
        self.inputReshape = (16, 16)
        super(blockCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            #nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, len(os.listdir(self.trainpath))),
            nn.Dropout(p=0.5)
        )
        # Multihead attention allows the model to understand spatial aspects of an image
        # Unsure if this is useful for blocks
        self.mha = nn.MultiheadAttention(embed_dim=3, num_heads=3, batch_first=True)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=8)

    def forward(self, x):
        batchSize, channels, h, w = x.shape
        x = x.view(batchSize, channels, h*w)
        x = x.permute(0, 2, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(0, 2, 1)
        x = x.view(batchSize, channels, h, w)
        x = self.features(x)
        return x


class flattenDataset(Dataset):
    """
        Builds a dataset that takes in dir like this with root dir as training:
        training /
            category1 /
                group1 /
                    files...
                group2 /
                    files...
            category2 /
                group1 /
                    files...
                group2 /
                    files...
                group3 /
                    files...
        And makes it be read like this:
        training /
            category1 /
                files...
            category2 /
                files...
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.categories = sorted([directory for directory in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, directory))])
        self.catalog = {category: self.recursivelyGetFilesInDir(os.path.join(root_dir, category)) for category in self.categories}
        self.samples = []
        for category in self.categories:
            for file in self.recursivelyGetFilesInDir(os.path.join(root_dir, category)):
                self.samples.append((file, category))

    def recursivelyGetFilesInDir(self, directory):
        files = []
        for path in os.listdir(directory):
            subPath = os.path.join(directory, path)
            if os.path.isdir(subPath):
                files.extend(self.recursivelyGetFilesInDir(subPath))
            else:
                files.append(subPath)
        return files

    def __len__(self):
        return sum(len(value) for value in self.catalog.values())

    @functools.cache
    def __getitem__(self, index):
        imagePath, category = self.samples[index]
        image = Image.open(imagePath)
        #image = cv2.imread(imagePath)
        if self.transform is not None: image = self.transform(image)
        if self.target_transform is not None: image = self.target_transform(image)
        categoryIndex = self.categories.index(category)
        return image, categoryIndex


class multiHeadDataset(Dataset):
    def __init__(self, root_dir='./training/tile', transform=None):
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

    @functools.cache
    def __getitem__(self, idx):
        imagePath, categoryIndex, ID = self.samples[idx]
        #image = Image.open(imagePath)
        image = cv2.imread(imagePath)
        #if random.random() < 0.01: cv2.imshow(str(idx), image)
        if self.transform is not None: image = self.transform(image)
        id_idx = self.id2idx[ID]
        return image, id_idx  # image, categoryIndex, id_idx


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
        net = blockCNN(trainpath='./training')
        #net.calculateNormalizationData()
        net.load()
        # Below is for Colab training
        #net.savepath = './content/drive/MyDrive/terrabot/nn/tile.cnn'
        net = torch.compile(net)  # backend="cudagraphs")
        net.trainModel()
        exit(0)
    if sys.argv[1] in ['tile', 'air', 'wall', 'liquid']:
        net = blockCNN(trainpath=f'./training/{sys.argv[1]}')
        net.load()
        net.trainModel()
    elif sys.argv[1] == 'offset':
        net = offsetCNN()
        net.load()
        net.trainModel()
