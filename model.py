import os
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from google.colab import drive
drive.mount('/content/drive')

#====================>ResNet Model<=====================
# Basic Residual Block for ResNet-38, ResNet-50 and ResNet-110
class BasicBlock(nn.Module):
    expansion = 1

    # Workflow input->conv1->bn1->relu->conv2->bn2->add->relu return the final result
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

# ResNet Model with dynamic layers
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3,
                 initial_channels=64, channel_scaling=2):
        super().__init__()
        self.in_channels = initial_channels
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Dynamically build layers which able to pass either 3 block layer or 4
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(layers):
            out_channels = initial_channels * (channel_scaling ** i)
            stride = 2 if i > 0 else 1  # Downsample after the first layer
            layer = self._make_layer(block, out_channels, num_blocks, stride)
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels * (channel_scaling ** (len(layers) - 1)) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    # create a residual layer with multiple basic blocks
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ========================>data loading for training and testing<==========================
# Normalization stats for CIFAR-10 which compute from raw data at training dataset
data_statistics = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

# implement the data augumentation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomRotation(10),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.GaussianBlur(kernel_size=(3,3)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value=1.0, inplace=False),
        transforms.Normalize(*data_statistics, inplace=True)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*data_statistics, inplace=True)
    ])
    return train_transform, test_transform

# For Creating proper dataset
class CIFAR10CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

# get train and eval dataloaders
def get_train_and_eval_dataloaders(batch_size=128, data_dir='/content/drive/MyDrive/DL/deep-learning-spring-2025-project-1/cifar-10-batches-py/'):
    train_transform, test_transform = get_transforms()

    meta_data_dict = load_cifar_batch(os.path.join(data_dir, 'batches.meta'))
    label_names = meta_data_dict[b'label_names']  

    # Load training batches
    train_data_list = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = load_cifar_batch(batch_file)
        batch_data = batch_dict[b'data']
        batch_data = batch_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        train_data_list.append(batch_data)
        train_labels += batch_dict[b'labels']
    train_images = np.concatenate(train_data_list, axis=0)
    train_dataset = CIFAR10CustomDataset(train_images, train_labels, transform=train_transform)

    # Load test batch
    test_batch = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    test_images = test_batch[b'data']
    test_images = test_images.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = test_batch[b'labels']
    eval_dataset = CIFAR10CustomDataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, eval_loader

# find available GPUs, if not, using cpu
def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#==================>Model Training <==================
# Training the model
def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    # Step the scheduler if it's not ReduceLROnPlateau
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    return avg_loss, accuracy

# evaluate the model
def evaluate(model, loader, criterion, device, scheduler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

#==================>ResNet set functions<==================
# Creating ResNet Archetecture
def ResNetCustom(block_type=BasicBlock,num_classes=10, in_channels=3, initial_channels=32, channel_scaling=2, layers=None):
    if layers is None:
        layers = [2, 3, 3, 2]
    return ResNet(block_type, layers, num_classes=num_classes,
                  in_channels=in_channels, initial_channels=initial_channels,
                  channel_scaling=channel_scaling)

# Find total parameters for that model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Scheduler setting function
def setup_scheduler(optimizer, scheduler_type, num_epochs):
    """
    Returns a learning rate scheduler based on the specified type.
    """
    if scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140], gamma=0.1)
    elif scheduler_type == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    else:
        return None
    
#==================> model training <==================
def train_model(model, train_loader, eval_loader, device, criterion, optimizer, scheduler, num_epochs, checkpoint_dir):
    best_eval_acc = 0.0
    results = []

    # Define warm-up parameters
    warmup_epochs = 15  # Number of warm-up epochs
    initial_lr = 0.01  # Start with lower learning rate
    target_lr = 0.1  # Standard learning rate

    for epoch in range(num_epochs):
        # Apply warm-up for the first few epochs
        if epoch < warmup_epochs:
            new_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Warm-up Epoch [{epoch+1}/{warmup_epochs}] - Adjusted LR: {new_lr:.6f}")
        elif scheduler is not None:
            scheduler.step()  # Apply scheduler after warm-up
            # scheduler.step(validation_metric)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)

        # Evaluate model
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device, scheduler)

        results.append((epoch + 1, train_loss, train_acc, eval_loss, eval_acc))

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")

        # Save checkpoint if the current evaluation accuracy is the best so far
        best_eval_acc = save_checkpoint(model, epoch, eval_acc, best_eval_acc, checkpoint_dir)

    return results, best_eval_acc

#==================> saving & test_unlabel dataset <==================
# loading unlable data
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, idx  # Return index as the image ID.

def load_custom_test_data(pkl_file, transform=None,batch_size=126):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    print("Keys in custom test data:")  # Debug: print available keys

    images = data[b'data']
    test_dataset = TestDataset(images, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# creating csv file
def generate_submission(model, loader, device, out_csv=None):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())

    df = pd.DataFrame({"ID": range(len(all_preds)), "Labels": all_preds})

    # If not provided, build our default path:
    if out_csv is None:
        output_dir = "/content/drive/MyDrive/DL/output"
        os.makedirs(output_dir, exist_ok=True)  # Create if doesn't exist
        out_csv = os.path.join(output_dir, "lastfile.csv")

    df.to_csv(out_csv, index=False)
    print(f"Submission saved to {out_csv}")

# saving checkpoint
def save_checkpoint(model, epoch, eval_acc, best_eval_acc, checkpoint_dir):
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        checkpoint_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} with Eval Acc: {eval_acc:.2f}%")
    return best_eval_acc

#==================> Main <==================
def main():
    # Configuration parameters
    num_epochs = 1
    batch_size = 256
    data_dir = "/content/drive/MyDrive/DL/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py"
    checkpoint_dir = "/content/drive/MyDrive/DL/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device and data loaders setup
    device = get_default_device()
    print("Using device:", device)
    train_loader, eval_loader = get_train_and_eval_dataloaders(batch_size, data_dir)

    # Model initialization and configuration
    model = ResNetCustom(block_type=BasicBlock, layers=[18,18,18], initial_channels=16).to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)

    # Loss, optimizer, and scheduler setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_type = "MultiStepLR"  # Choose scheduler type here
    scheduler = setup_scheduler(optimizer, scheduler_type, num_epochs)

    # Check model parameter count
    if count_parameters(model) > 5000000:
        print("Model has too many parameters:", count_parameters(model))
        return
    print("Total Parameters:", count_parameters(model))

    # # Train the model and save checkpoints when improvements occur
    results, best_eval_acc = train_model(model, train_loader, eval_loader, device, criterion, optimizer, scheduler, num_epochs, checkpoint_dir)

    # Save training history and generate plots
    df = pd.DataFrame(results, columns=['Epoch', 'Train Loss', 'Train Acc', 'Eval Loss', 'Eval Acc'])
    df.to_csv('/content/drive/MyDrive/DL/results/layer4[18,18,18]_with_weight_decay_epoch_vs_loss.csv', index=False)
    df.plot(x='Epoch', y=['Train Loss', 'Eval Loss'], title='Epochs vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('epoch_vs_loss[18,18,18].png')
    plt.show()

    # Save final model and generate submission CSV
    torch.save(model.state_dict(), 'Resnet110(4).pth')
    custom_test_file = "/content/drive/MyDrive/DL/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl"
    _, test_transform = get_transforms()
    custom_test_loader = load_custom_test_data(custom_test_file, transform=test_transform, batch_size=batch_size)
    generate_submission(model, custom_test_loader, device, out_csv="submission[18,18,18]4.csv")

if __name__ == "__main__":
    main()