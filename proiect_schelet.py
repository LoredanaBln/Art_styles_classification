import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Configuration
EPOCHS = 80 # cate treceri intregi prin tot datasetul
BATCH_SIZEg = 8
NUM_CLASSES = 6
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your dataset
dataset_path = "E:\\AN3\\SEM2\\si\\art_style_classification\\dataset"


def get_image_label_pairs(dataset_path): # function that gets pairs of image and label
    # TODO: BETTER TO COMPLETE WITH YOUR CODE, in functie de cum aveti downloadat
    samples = []
    for class_label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                samples.append((img_path, int(class_label)))
    return samples


# Custom Dataset
class HandGestureDataset(Dataset):
    def __init__(self, samples, image_size=(128, 128)):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img):
        # TODO: toate imaginile trebuie sa aiba acelasi size
        # TODO:NORMALIZARE: aici ar trebui sa normalizati cu mean si std dev
        img = np.expand_dims(img, axis=0)  # Convert to (1, H, W)
        return torch.from_numpy(img) # se numesc tensori -> transformam din numpy in torch ca asa avem nevoie

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.preprocess_image(img)
        return img, label

# Prepare dataset
all_samples = get_image_label_pairs(dataset_path)
# TODO: IN FUNCTIE DE MARIMEA DATASETULUI MODIFICATI PROPORTIA PT VAL (also daca aveti destule date impartititi in 3)
train_samples, val_samples = train_test_split(all_samples, test_size=0.25, random_state=42)

# Datasets and Dataloaders
train_dataset = HandGestureDataset(train_samples, image_size=IMAGE_SIZE)
val_dataset = HandGestureDataset(val_samples, image_size=IMAGE_SIZE)

# in teorie trebuie train test val, daca nu aveti destule date e okay si train val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last = true foaret important
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # la validare neaparat shuffle false


class HandGestureCNN(nn.Module): # reteta e conv batch relu
    def __init__(self, num_classes=NUM_CLASSES):
        super(HandGestureCNN, self).__init__()
        # COMPLETE WITH YOUR CODE

    def forward(self, x):
        # COMPLETE WITH YOUR CODE
        pass


model = HandGestureCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()  # orice alt loss ex: KLDivLoss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,) # voi va trebui sa folositi si SGD si SGD cu momentum

# Training loop (unchanged)
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_loss = float('inf')
    for epoch in range(epochs):

        model.train() #### IMPORTANT

        running_loss, total = 0.0, 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) ### IMPORTANT
            # asta e reteta de antrenare
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            # COMPLETE WITH YOUR CODE FOR METRICS

        epoch_loss = running_loss / total

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: %")

        model.eval() 

        val_loss, val_total = 0.0, 0

        with torch.no_grad(): #### IMPORTANT
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                # COMPLETE WITH YOUR CODE FOR METRICS
        val_epoch_loss = val_loss / val_total
        
        print(f"Validation Loss: {val_epoch_loss:.4f} Accuracy: %")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), f"./models/predat_model_best.pth")
            print("Model saved!")

train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
torch.save(model.state_dict(), "./models/codiax_model_final.pth")
