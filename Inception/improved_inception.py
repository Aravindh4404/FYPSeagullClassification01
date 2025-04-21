import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

class InceptionV3Modified(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        Modified Inception v3 model for binary classification.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for binary)
            dropout_rate (float): Dropout rate for regularization (default: 0.5)
        """
        super(InceptionV3Modified, self).__init__()
        # Load pre-trained Inception v3 with ImageNet weights
        from torchvision.models import Inception_V3_Weights
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        
        # Get number of features from the final layer
        num_ftrs = self.inception.fc.in_features
        
        # Replace the final fully connected layer with a custom classifier
        self.inception.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for the new layers
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the new layers using Xavier initialization"""
        for m in self.inception.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            During training: (aux_output, main_output)
            During inference: main_output
        """
        if self.training:
            # Return both auxiliary and main outputs during training
            aux, x = self.inception(x)
            return aux, x
        else:
            # Return only main output during inference
            return self.inception(x)

def create_model(device, num_classes=2, dropout_rate=0.5):
    """
    Create and initialize the Inception v3 model.
    
    Args:
        device (torch.device): Device to place the model on
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        model: Initialized InceptionV3Modified model
        criterion: Loss function
        optimizer: Optimizer with L2 regularization
        scheduler: Learning rate scheduler
    """
    # Initialize the model
    model = InceptionV3Modified(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer with L2 regularization
    optimizer = optim.AdamW(model.parameters(), 
                           lr=0.0001146316711604839, 
                           weight_decay=0.0009719766132928495)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   mode='max', 
                                                   factor=0.1, 
                                                   patience=3)
    
    return model, criterion, optimizer, scheduler 