import sys
sys.path.append('../src')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from vit_model import VisionTransformer
from preprocessing import prepare_dl_data
 
data_path = '../data_dl/train'
X, y = prepare_dl_data(data_path)
print(f'Data shape: {X.shape}, Labels: {y.shape}')
print(f'Class distribution: {np.bincount(y)}')

from typing import Any

class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = indices[:split], indices[split:]

train_dataset = ImageDataset(X[train_idx], y[train_idx])
val_dataset = ImageDataset(X[val_idx], y[val_idx])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

 
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformer(img_size=224, patch_size=16, num_classes=2, 
                          embed_dim=384, depth=6, num_heads=6)
model = model.to(device)
print(f'Model on {device}')
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    batch_count = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += y_batch.size(0)
        train_correct += predicted.eq(y_batch).sum().item()
        batch_count += 1
        if batch_count % 100 == 0:
            print(f'  Batch {batch_count}/{len(train_loader)}')
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += y_batch.size(0)
            val_correct += predicted.eq(y_batch).sum().item()
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {val_acc:.2f}%')
    
 
model.eval()
with torch.no_grad():
    X_test, y_test = next(iter(val_loader))
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    _, predicted = outputs.max(1)
    probs = torch.softmax(outputs, dim=1)
    
    print('Sample predictions:')
    for i in range(len(y_test)):
        true_label = 'cat' if y_test[i] == 0 else 'dog'
        pred_label = 'cat' if predicted[i] == 0 else 'dog'
        confidence = probs[i][predicted[i]].item() * 100
        print(f'True: {true_label}, Pred: {pred_label}, Confidence: {confidence:.2f}%')

 
import timm
pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
pretrained_model = pretrained_model.to(device)
print(f'Pretrained model on {device}')
print(f'Total parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}')

 
criterion_pretrained = nn.CrossEntropyLoss()
optimizer_pretrained = torch.optim.AdamW(pretrained_model.parameters(), lr=1e-4, weight_decay=0.01)

epochs_pretrained = 3
for epoch in range(epochs_pretrained):
    pretrained_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    batch_count = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_pretrained.zero_grad()
        outputs = pretrained_model(X_batch)
        loss = criterion_pretrained(outputs, y_batch)
        loss.backward()
        optimizer_pretrained.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += y_batch.size(0)
        train_correct += predicted.eq(y_batch).sum().item()
        batch_count += 1
        if batch_count % 100 == 0:
            print(f'  Batch {batch_count}/{len(train_loader)}')
    
    pretrained_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = pretrained_model(X_batch)
            loss = criterion_pretrained(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += y_batch.size(0)
            val_correct += predicted.eq(y_batch).sum().item()
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    print(f'PRETRAINED Epoch {epoch+1}/{epochs_pretrained} - Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {val_acc:.2f}%')

print(train_acc)
print(val_acc)