import torch
import matplotlib.pyplot as plt

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

def pytorch_pred_test_tta(model, test_loader, device, name='submission.csv'):
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

def adjust_data(x_train, x_valid, x_test):
    x_train_raw = (x_train + 1) / 2.0
    x_valid_raw = (x_valid + 1) / 2.0
    x_test_raw  = (x_test + 1) / 2.0

    calculated_mean = x_train_raw.mean(axis=(0, 1, 2))
    calculated_std = x_train_raw.std(axis=(0, 1, 2))

def evaluate(model):
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