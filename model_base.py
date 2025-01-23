import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union
import os

class BaseModel(nn.Module, ABC):
    """Base class for all neural network models in the snake AI project.
    
    This abstract class defines the common interface that all models must implement,
    and provides shared functionality for device management.
    """
    _default_device: Optional[torch.device] = None

    def __init__(self, device: Optional[Union[torch.device, str]] = None) -> None:
        """Initialize the base model.
        
        Args:
            device: Device to use. Can be 'cpu', 'mps', or None (auto-detect).
                   Also accepts torch.device objects. If None, will use the default
                   device or auto-detect based on system capabilities.
        """
        super().__init__()
        # Initialize device before any other operations
        if device is not None:
            self.device = torch.device(device)
        elif self._default_device is not None:
            self.device = self._default_device
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Move model to device immediately
        self.to(self.device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass

    def save(self, file_name: str) -> None:
        """Save the model's state to a file.
        
        Args:
            file_name: Name of the file to save the model to
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        """Load the model's state from a file.
        
        Args:
            file_name: Name of the file to load the model from
        """
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        state_dict = torch.load(file_name, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict)
        self.to(self.device)  # Ensure model is on correct device after loading
