import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F

base_log_dir = f'runs/art_style_classification_'

writers = {
    'optimizers': SummaryWriter(f'{base_log_dir}/optimizers'),
    'batch_sizes': SummaryWriter(f'{base_log_dir}/batch_sizes'),
    'learning_rates': SummaryWriter(f'{base_log_dir}/learning_rates'),
    'loss_functions': SummaryWriter(f'{base_log_dir}/loss_functions'),
    'cross_validation': SummaryWriter(f'{base_log_dir}/cross_validation'),
    'statistics': SummaryWriter(f'{base_log_dir}/statistics')
}

os.makedirs('./models', exist_ok=True)

# Configuration
EPOCHS = 30 # cate treceri intregi prin tot datasetul
BATCH_SIZE = 16
NUM_CLASSES = 6
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['Abstract_Expressionism', 'Baroque', 'Contemporary_Realism',
               'Early_Renaissance', 'Fauvism', 'Symbolism']

# Define the path to your dataset
dataset_path = "E:\\AN3\\SEM2\\si\\art_style_classification\\dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "validation")
test_path = os.path.join(dataset_path, "test")

def get_image_label_pairs(dataset_path):
    samples = []
    class_counts = {}
    
    class_mapping = {
        'Abstract_Expressionism': 0,
        'Baroque': 1,
        'Contemporary_Realism': 2,
        'Early_Renaissance': 3,
        'Fauvism': 4,
        'Symbolism': 5
    }
    
    for class_name in class_mapping:
        class_counts[class_mapping[class_name]] = 0

    for class_name in class_mapping:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    class_label = class_mapping[class_name]
                    samples.append((img_path, class_label))
                    class_counts[class_label] += 1

    return samples, class_counts

# dataset custom
class HandGestureDataset(Dataset):
    def __init__(self, samples, image_size=(128, 128), is_training=False):
        self.samples = samples
        self.image_size = image_size
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img):
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)

        if self.is_training:
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomRotation(10)(img)
            img = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(img)

        img = transforms.ToTensor()(img)
        mean = img.mean()
        std = img.std()
        img = transforms.Normalize(mean=[mean], std=[std])(img)

        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.preprocess_image(img)
        return img, label

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    # acuratete
    accuracy = correct / total
    
    # recall pentru fiecare clasa
    recalls = []
    for i in range(NUM_CLASSES):
        true_positives = ((predicted == i) & (labels == i)).sum().item()
        actual_positives = (labels == i).sum().item()
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recalls.append(recall)
    
    # entropie pentru fiecare clasa
    probs = F.softmax(outputs, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    
    return accuracy, recalls, entropy.item(), predicted

def log_metrics(writer, metrics, epoch, phase, config_name):
    accuracy, recalls, entropy, predicted = metrics
    
    writer.add_scalar(f'{config_name}/{phase}/accuracy', accuracy, epoch)
    writer.add_scalar(f'{config_name}/{phase}/entropy', entropy, epoch)
    
    for i, (recall, class_name) in enumerate(zip(recalls, CLASS_NAMES)):
        writer.add_scalar(f'{config_name}/{phase}/class_{class_name}/recall', recall, epoch)
    
    return accuracy, recalls, entropy

def log_confusion_matrix(writer, cm, class_names, epoch, config_name):
    # matrice de confuzie
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {config_name}')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm_normalized.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    image = buf[:, :, :3]
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    writer.add_image(f'{config_name}/confusion_matrix', image, epoch)
    plt.close(figure)
    
    for i, class_name in enumerate(class_names):
        precision = cm[i, i] / (cm[:, i].sum() + 1e-8)
        recall = cm[i, i] / (cm[i, :].sum() + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        writer.add_scalar(f'{config_name}/class_{class_name}/precision', precision, epoch)
        writer.add_scalar(f'{config_name}/class_{class_name}/recall', recall, epoch)
        writer.add_scalar(f'{config_name}/class_{class_name}/f1', f1, epoch)

def visualize_dataset_statistics(samples, class_counts, title, writer):
    class_distribution = {CLASS_NAMES[i]: count for i, count in class_counts.items()}
    writer.add_scalars(f'dataset_stats/{title}/class_distribution', class_distribution, 0)
    
    # histograme pentru fiecare clasa pentru canalele RGB
    for class_id in range(NUM_CLASSES):
        class_samples = [(img_path, label) for img_path, label in samples if label == class_id]
        
        r_hist = np.zeros(256)
        g_hist = np.zeros(256)
        b_hist = np.zeros(256)
        
        for img_path, _ in class_samples[:140]:
            img = cv2.imread(img_path)
            if img is not None:
                # Split into RGB channels
                b, g, r = cv2.split(img)
                
                # Calculate histograms
                r_hist += cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
                g_hist += cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
                b_hist += cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
        
        total_pixels = np.sum(r_hist)
        if total_pixels > 0:
            r_hist = r_hist / total_pixels
            g_hist = g_hist / total_pixels
            b_hist = b_hist / total_pixels
        
        writer.add_histogram(f'dataset_stats/{title}/class_{CLASS_NAMES[class_id]}/red_channel', r_hist, 0, max_bins=256)
        writer.add_histogram(f'dataset_stats/{title}/class_{CLASS_NAMES[class_id]}/green_channel', g_hist, 0, max_bins=256)
        writer.add_histogram(f'dataset_stats/{title}/class_{CLASS_NAMES[class_id]}/blue_channel', b_hist, 0, max_bins=256)

# implementare model cu cnn - cerinta 3
class HandGestureCNN(nn.Module): # reteta e conv batch relu
    def __init__(self, num_classes=NUM_CLASSES):
        super(HandGestureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # ca sa reduc dimensiunea la jumatate

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# experimenteaza cu functii de loss diferite - cerinta 7
class KLDivLossWrapper(nn.Module):
    def __init__(self):
        super(KLDivLossWrapper, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels):
        # Convert outputs to log probabilities
        log_probs = F.log_softmax(outputs, dim=1)

        # Convert labels to one-hot encoded probabilities
        target_probs = F.one_hot(labels, num_classes=outputs.size(1)).float()

        return self.kl_div(log_probs, target_probs)


# initializeaza model
model = HandGestureCNN().to(DEVICE)

loss_functions = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'kullback_leibler': KLDivLossWrapper()
}

# experimenteaza cu optimizatori diferiti - cerinta 6
optimizers = {
    'adam': optim.Adam(model.parameters(), lr=LEARNING_RATE),
    'sgd': optim.SGD(model.parameters(), lr=LEARNING_RATE),
    'sgd_momentum': optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9),
}

# experimenteaza cu batch size-uri diferite - cerinta 4
batch_sizes = {
    'small': 8,
    'medium': 16,
    'large': 32
}

# experimenteaza cu learning rate-uri diferite - cerinta 5
learning_rates = {
    'small': 1e-4,
    'medium': 1e-3,
    'large': 1e-2
}

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['adam'], mode='min', factor=0.1, patience=5)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, config_name, experiment_type):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    writer = writers[experiment_type]
    
    sample_input = next(iter(train_loader))[0].to(DEVICE)
    writer.add_graph(model, sample_input)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        all_train_outputs = []
        all_train_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'{config_name}/gradients/{name}', param.grad, epoch)
            
            optimizer.step()
            
            running_loss += loss.item()
            all_train_outputs.append(outputs)
            all_train_labels.append(labels)
        
        # training metrics
        train_outputs = torch.cat(all_train_outputs)
        train_labels = torch.cat(all_train_labels)
        train_metrics = calculate_metrics(train_outputs, train_labels)
        train_accuracy, train_recalls, train_entropy = log_metrics(writer, train_metrics, epoch, 'train', config_name)
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # validation
        model.eval()
        val_loss = 0.0
        all_val_outputs = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_val_outputs.append(outputs)
                all_val_labels.append(labels)
        
        # validation metrics
        val_outputs = torch.cat(all_val_outputs)
        val_labels = torch.cat(all_val_labels)
        val_metrics = calculate_metrics(val_outputs, val_labels)
        val_accuracy, val_recalls, val_entropy = log_metrics(writer, val_metrics, epoch, 'val', config_name)
        
        # confusion matrix
        cm = confusion_matrix(val_labels.cpu().numpy(), val_metrics[3].cpu().numpy())
        log_confusion_matrix(writer, cm, CLASS_NAMES, epoch, config_name)
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # learning rate
        scheduler.step(epoch_val_loss)
        
        # losses and learning rate
        writer.add_scalar(f'{config_name}/train_loss', epoch_train_loss, epoch)
        writer.add_scalar(f'{config_name}/val_loss', epoch_val_loss, epoch)
        writer.add_scalar(f'{config_name}/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # parameter histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f'{config_name}/parameters/{name}', param.data, epoch)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print('Validation Recalls:', [f'{r:.4f}' for r in val_recalls])
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'./models/best_model_{experiment_type}_{config_name}.pth')
    
    return train_losses, val_losses, train_accuracy, val_accuracy

def test_model(model, test_loader, criterion, config_name):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_accuracy += (predicted == labels).sum().item() / labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    # test metrics
    writers['cross_validation'].add_scalar(f'{config_name}/test_loss', test_loss, 0)
    writers['cross_validation'].add_scalar(f'{config_name}/test_accuracy', test_accuracy, 0)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    log_confusion_matrix(writers['cross_validation'], cm, CLASS_NAMES, 0, f'{config_name}_test')

    print(f'\nTest Results for {config_name}:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_loss, test_accuracy

def run_optimizer_experiments(train_dataset, val_dataset):
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name} optimizer")
        model = HandGestureCNN().to(DEVICE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, opt_name, 'optimizers')


def run_batch_size_experiments(train_dataset, val_dataset):
    for batch_name, batch_size in batch_sizes.items():
        print(f"\nTraining with {batch_name} batch size")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        model = HandGestureCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, batch_name, 'batch_sizes')


def run_learning_rate_experiments(train_dataset, val_dataset):
    for lr_name, lr in learning_rates.items():
        print(f"\nTraining with {lr_name} learning rate")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        model = HandGestureCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, lr_name, 'learning_rates')

def run_loss_function_experiments(train_dataset, val_dataset):
    for loss_name, criterion in loss_functions.items():
        print(f"\nTraining with {loss_name} loss function")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        model = HandGestureCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, loss_name, 'loss_functions')

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, fold):
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item() / labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        
        writers['cross_validation'].add_scalar(f'fold_{fold+1}/train_loss', train_loss, epoch)
        writers['cross_validation'].add_scalar(f'fold_{fold+1}/train_accuracy', train_acc, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == labels).sum().item() / labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)
        
        writers['cross_validation'].add_scalar(f'fold_{fold+1}/val_loss', val_loss, epoch)
        writers['cross_validation'].add_scalar(f'fold_{fold+1}/val_accuracy', val_acc, epoch)
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./models/best_model_fold_{fold+1}.pth')
            
            cm = confusion_matrix(all_labels, all_preds)
            log_confusion_matrix(writers['cross_validation'], cm, CLASS_NAMES, epoch, f'fold_{fold+1}')
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_acc': max(fold_val_accs),
        'train_losses': fold_train_losses,
        'val_losses': fold_val_losses,
        'train_accs': fold_train_accs,
        'val_accs': fold_val_accs
    }

def cross_validate(model_class, dataset, num_folds=5, batch_size=BATCH_SIZE, epochs=EPOCHS):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nFOLD {fold + 1}/{num_folds}')
        
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        model = model_class().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()
        
        fold_result = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, fold)
        fold_results.append(fold_result)
    
    # cross-validation results
    avg_val_loss = np.mean([result['best_val_loss'] for result in fold_results])
    avg_val_acc = np.mean([result['best_val_acc'] for result in fold_results])
    std_val_loss = np.std([result['best_val_loss'] for result in fold_results])
    std_val_acc = np.std([result['best_val_acc'] for result in fold_results])
    
    writers['cross_validation'].add_scalar('cv_results/avg_val_loss', avg_val_loss, 0)
    writers['cross_validation'].add_scalar('cv_results/avg_val_acc', avg_val_acc, 0)
    writers['cross_validation'].add_scalar('cv_results/std_val_loss', std_val_loss, 0)
    writers['cross_validation'].add_scalar('cv_results/std_val_acc', std_val_acc, 0)
    
    print('\nCross-validation Results:')
    print(f'Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}')
    print(f'Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}')
    
    return fold_results

if __name__ == "__main__":
    train_samples, train_class_counts = get_image_label_pairs(train_path)
    val_samples, val_class_counts = get_image_label_pairs(val_path)
    test_samples, test_class_counts = get_image_label_pairs(test_path)

    visualize_dataset_statistics(train_samples, train_class_counts, 'train', writers['statistics'])
    visualize_dataset_statistics(val_samples, val_class_counts, 'validation', writers['statistics'])
    visualize_dataset_statistics(test_samples, test_class_counts, 'test', writers['statistics'])

    train_dataset = HandGestureDataset(train_samples, image_size=IMAGE_SIZE, is_training=True)
    val_dataset = HandGestureDataset(val_samples, image_size=IMAGE_SIZE, is_training=False)
    test_dataset = HandGestureDataset(test_samples, image_size=IMAGE_SIZE, is_training=False)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    run_optimizer_experiments(train_dataset, val_dataset)
    # run_batch_size_experiments(train_dataset, val_dataset)
    # run_learning_rate_experiments(train_dataset, val_dataset)
    # run_loss_function_experiments(train_dataset, val_dataset)

    # all_samples = train_samples + val_samples
    # all_dataset = HandGestureDataset(all_samples, image_size=IMAGE_SIZE, is_training=False)
    # print("Starting K-fold Cross Validation...")
    # cv_results = cross_validate(HandGestureCNN, all_dataset, num_folds=5, batch_size=BATCH_SIZE, epochs=EPOCHS)
