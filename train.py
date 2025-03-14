import torch
import torch.optim as optim
import torch.nn as nn
import torch_directml 
from data_processing import get_dataloaders
from model import get_model
from config import (
    BATCH_SIZE, EPOCHS, DATA_DIR, DEVICE, LR, WEIGHT_DECAY, GRADIENT_CLIP,
    DROPOUT_RATE, LABEL_SMOOTHING, EARLY_STOPPING_PATIENCE, BEST_MODEL_PATH,
    FINAL_MODEL_PATH, MIXUP_ALPHA
)
import time
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Apply mixup with probability 0.5"""
    if alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    return x, y, y, 1.0

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def unfreeze_model_layers(model, epoch):
    """Gradually unfreeze EfficientNet layers as training progresses"""
    if epoch == 5:  # After 5 epochs of training the classifier
        # Unfreeze the last convolutional block
        for name, param in model.named_parameters():
            if "features.8" in name:  # The last block in EfficientNet
                param.requires_grad = True
        print("Unfreezing last convolutional block")
    elif epoch == 10:  # After 10 epochs, unfreeze more layers
        # Unfreeze the last two blocks
        for name, param in model.named_parameters():
            if "features.7" in name or "features.8" in name:
                param.requires_grad = True
        print("Unfreezing second-to-last convolutional block")


def train(binary=True):
    """
    Train the model for either binary or multi-class classification
    
    Args:
        binary (bool): If True, train for binary classification (fresh vs rotten)
    """
    # DirectML device
    dml_device = torch_directml.device()
    
    # Get data loaders with binary flag
    train_loader, test_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE, binary=binary)

    # Initialize model with dropout and binary flag
    model = get_model(num_classes, dropout_rate=DROPOUT_RATE, binary=binary).to(dml_device)
    
    # Calculate class weights
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(dml_device)
    
    # Use weighted loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # Lower initial learning rate
        weight_decay=0.01,  # Increased weight decay
        eps=1e-8
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    best_acc = 0.0
    patience_counter = 0
    
    # Training loop
    start_time = time.time()
    for epoch in range(EPOCHS):
        # Progressive unfreezing
        unfreeze_model_layers(model, epoch)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            images = images.to(dml_device, non_blocking=True)
            labels = labels.to(dml_device, non_blocking=True)
            
            # Apply mixup
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(mixed_images)
            
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # For accuracy calculation, use original images without mixup
            with torch.no_grad():
                outputs_orig = model(images)
                _, predicted = outputs_orig.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Calculate training metrics
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validation"):
                images = images.to(dml_device, non_blocking=True)
                labels = labels.to(dml_device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate validation metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted'
        )
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, BEST_MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
    }, FINAL_MODEL_PATH)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train(binary=True)  # Set to True for binary classification