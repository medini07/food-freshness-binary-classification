import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    DenseNet121_Weights
)
from config import MODEL_NAME, PRETRAINED

def get_model(num_classes, dropout_rate=0.5, binary=False):
    """
    Get a pre-trained model with custom classifier.
    
    Args:
        num_classes (int): Number of output classes (ignored if binary=True)
        dropout_rate (float): Dropout rate for the classifier layers
        binary (bool): If True, create a binary classifier (fresh vs rotten)
    
    Returns:
        model: The configured model
    """
    output_classes = 2 if binary else num_classes
    
    if MODEL_NAME == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if PRETRAINED else None
        model = models.efficientnet_b2(weights=weights)
        
        # First only train the classifier
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Enhanced classifier with more capacity
        num_features = model.classifier[1].in_features
        if binary:
            # Simplified classifier for binary classification
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(256, output_classes)
            )
        else:
            # Full classifier for multi-class
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, output_classes)
            )
        
    elif MODEL_NAME == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if PRETRAINED else None
        model = models.resnet50(weights=weights)
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
            
        # Enhanced classifier
        num_features = model.fc.in_features
        if binary:
            # Simplified classifier for binary classification
            model.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(256, output_classes)
            )
        else:
            # Full classifier for multi-class
            model.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, output_classes)
            )
        
    elif MODEL_NAME == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if PRETRAINED else None
        model = models.densenet121(weights=weights)
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
            
        # Enhanced classifier
        num_features = model.classifier.in_features
        if binary:
            # Simplified classifier for binary classification
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(256, output_classes)
            )
        else:
            # Full classifier for multi-class
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate/2),
                nn.Linear(512, output_classes)
            )
    
    else:
        raise ValueError(f"Model {MODEL_NAME} not supported")
    
    return model
# Add this to model.py
def get_hierarchical_model(num_classes):
    """Create a model with hierarchical classification"""
    base_model = get_model(2)  # First classify fresh vs rotten
    
    # Then classify the specific fruit/vegetable
    num_fruits = num_classes // 2  # Assuming equal number of fresh/rotten classes
    fruit_model = get_model(num_fruits)
    
    return base_model, fruit_model