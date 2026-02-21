
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.sign_model import ASLClassifier, ASLDynamicClassifier
from src.data.dataset import ASLDataset, ASLDynamicDataset
from src.utils.metrics import calculate_metrics


def _train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, save_path, model_name):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y_batch.numpy())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        metrics = calculate_metrics(all_labels, all_preds)
        
        print(f"[{model_name}] Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best {model_name} to {save_path}")


def train_static(epochs=100, batch_size=16, lr=0.001, save_dir="models"):
    """Train the MLP model on static signs (A-Z except J, Z)."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    data_path, labels_path = "data/X.npy", "data/y.npy"
    
    if not os.path.exists(data_path):
        print("No static training data found at data/X.npy. Skipping static model training.")
        return
    
    print("=== Training Static Sign Model (MLP) ===")
    full_dataset = ASLDataset(data_path=data_path, labels_path=labels_path)
    unique_labels = np.unique(full_dataset.labels)
    print(f"Found {len(full_dataset)} samples across {len(unique_labels)} classes")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = ASLClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    _train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, save_path, "Static MLP")


def train_dynamic(epochs=100, batch_size=16, lr=0.001, save_dir="models", seq_len=30):
    """Train the LSTM model on dynamic/motion signs (J, Z, and phrases)."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_dynamic_model.pth")
    data_path, labels_path = "data/X_dynamic.npy", "data/y_dynamic.npy"
    
    if not os.path.exists(data_path):
        print("No dynamic training data found at data/X_dynamic.npy. Skipping dynamic model training.")
        return
    
    print("\n=== Training Dynamic Sign Model (LSTM) ===")
    full_dataset = ASLDynamicDataset(data_path=data_path, labels_path=labels_path, seq_len=seq_len)
    unique_labels = np.unique(full_dataset.labels)
    # Always use 6 classes to match API: J, Z, Hello, Goodbye, Please, Thank You
    DYNAMIC_NUM_CLASSES = 6
    num_classes = max(DYNAMIC_NUM_CLASSES, int(unique_labels.max()) + 1)
    print(f"Found {len(full_dataset)} sequences across {len(unique_labels)} classes (model uses {num_classes} output classes)")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use smaller batch size for small datasets, and don't drop last
    effective_bs = min(batch_size, max(2, train_size // 2))
    train_loader = DataLoader(train_dataset, batch_size=effective_bs, shuffle=True, drop_last=len(train_dataset) > effective_bs)
    val_loader = DataLoader(val_dataset, batch_size=effective_bs)
    
    model = ASLDynamicClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    _train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, save_path, "Dynamic LSTM")


def train_model(epochs=100, batch_size=16, lr=0.001, save_dir="models"):
    """Train both static and dynamic models."""
    train_static(epochs=epochs, batch_size=batch_size, lr=lr, save_dir=save_dir)
    train_dynamic(epochs=epochs, batch_size=batch_size, lr=lr, save_dir=save_dir)


if __name__ == "__main__":
    train_model(epochs=50)
