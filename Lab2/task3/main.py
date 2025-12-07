import copy 
import numpy as np
import pandas as pd

import data_generator
import imagedataset
import net

import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

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

BATCH_SIZE = 512
epochs = 40
learning_rate = 0.002 
L2_LAMBDA = 1e-4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

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

train_dataset = imagedataset.ImageDataset(x_train_t, y_train_t, transform=train_transforms)
valid_dataset = imagedataset.ImageDataset(x_valid_t, y_valid_t, transform=val_transforms)
test_dataset  = imagedataset.ImageDataset(x_test_t, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=6, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=6, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=6, pin_memory=True)

# Init Model
model = net.Net(num_classes=data.K).to(device)
criterion = net.nn.CrossEntropyLoss()
    
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