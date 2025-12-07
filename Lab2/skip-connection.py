# ==========================================
# 0. System & Graphics Fixes
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy 
from torchvision import transforms

# Import your local data generator
import data_generator

# ==========================================
# 1. Setup Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# ==========================================
# 2. Hyperparameters
# ==========================================
BATCH_SIZE = 128
epochs = 35
learning_rate = 0.001 
L2_LAMBDA = 1e-4 

# ==========================================
# 3. Data Preparation Class
# ==========================================
class ImageDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform:
            img = self.transform(img)
        if self.y is not None:
            return img, self.y[idx]
        return img

# (NOTE: Transforms removed from here because we need to calculate stats first!)

# ==========================================
# 4. Model Definition: "Wide & Shallow" ResNet
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.convs(x)
        shortcut = self.shortcut(x)
        out += shortcut      
        out = nn.LeakyReLU(0.1, inplace=True)(out)
        return out

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        # Initial Prep: Keep 32x32 resolution
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.block1 = ResidualBlock(32, 32, stride=1)
        self.block2 = ResidualBlock(32, 64, stride=2)
        self.block3 = ResidualBlock(64, 128, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.dense_block = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2), 
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.prep(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense_block(x)
        return x

# ==========================================
# 6. Evaluation Utils
# ==========================================
def pytorch_plot_training(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs_range = range(len(loss))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Train')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Train')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.title('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plt.savefig('training_plot.png')

def pytorch_pred_test_tta(model, test_loader, name='submission.csv'):
    model.eval()
    predictions = []
    print("Generating predictions using TTA (Original + Flip)...")
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            out1 = model(inputs)
            prob1 = torch.softmax(out1, dim=1)
            
            inputs_h = torch.flip(inputs, dims=[3])
            out2 = model(inputs_h)
            prob2 = torch.softmax(out2, dim=1)
            
            inputs_v = torch.flip(inputs, dims=[2])
            out3 = model(inputs_v)
            prob3 = torch.softmax(out3, dim=1)
            
            avg_prob = (prob1 + prob2 + prob3) / 3.0
            _, predicted = torch.max(avg_prob, 1)
            predictions.extend(predicted.cpu().numpy())
            
    df = pd.DataFrame({'class': predictions})
    df.index.name = 'id'
    df.to_csv(name)
    print(f'Done! Predictions saved to {name}.')

# ==========================================
# 7. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    print("Loading data from generator...")
    data = data_generator.DataGenerator()
    data.generate(dataset='patchcam')

    print("Adjusting data range from [-1, 1] to [0, 1] for PyTorch...")
    x_train_raw = (data.x_train + 1) / 2.0
    x_valid_raw = (data.x_valid + 1) / 2.0
    x_test_raw  = (data.x_test + 1) / 2.0

    # ======================================================
    # 1. Calculate Statistics
    # ======================================================
    print("Calculating specific statistics for this 32x32 dataset...")
    calculated_mean = x_train_raw.mean(axis=(0, 1, 2))
    calculated_std = x_train_raw.std(axis=(0, 1, 2))
    
    print(f"Computed MEAN: {calculated_mean}")
    print(f"Computed STD:  {calculated_std}")
    
    PCAM_MEAN = calculated_mean.tolist()
    PCAM_STD  = calculated_std.tolist()

    # ======================================================
    # 2. DEFINE TRANSFORMS HERE (Now that stats exist)
    # ======================================================
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        # Color Jitter
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        # Normalize using the variables we just calculated
        transforms.Normalize(mean=PCAM_MEAN, std=PCAM_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Normalize(mean=PCAM_MEAN, std=PCAM_STD)
    ])

    # ======================================================
    # 3. Create Datasets & Loaders
    # ======================================================
    x_train_t = torch.tensor(x_train_raw).permute(0, 3, 1, 2).float()
    x_valid_t = torch.tensor(x_valid_raw).permute(0, 3, 1, 2).float()
    x_test_t  = torch.tensor(x_test_raw).permute(0, 3, 1, 2).float()
    
    y_train_t = torch.tensor(data.y_train).long().squeeze()
    y_valid_t = torch.tensor(data.y_valid).long().squeeze()

    train_dataset = ImageDataset(x_train_t, y_train_t, transform=train_transforms)
    valid_dataset = ImageDataset(x_valid_t, y_valid_t, transform=val_transforms)
    test_dataset  = ImageDataset(x_test_t, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=6, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=6, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=6, pin_memory=True)

    # Init Model
    model = Net(num_classes=data.K).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_LAMBDA)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        scheduler.step(val_epoch_loss)
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"--> NEW BEST MODEL! (Acc: {best_acc:.4f})")
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.4f}")

    print(f"\nTraining Complete. Best Validation Acc: {best_acc:.4f}")
    
    # Reload Best Model
    model.load_state_dict(best_model_wts)
    
    # Plot and Predict
    pytorch_plot_training(history)
    pytorch_pred_test_tta(model, test_loader)