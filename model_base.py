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
                   Also accepts torch.device objects.
        """
        super().__init__()
        self._device = None
        if device is not None:
            self.device = device  # This will trigger the setter

    @classmethod
    def set_default_device(cls, device: Optional[Union[torch.device, str]] = None) -> None:
        """Set the default device for all new model instances.
        
        Args:
            device: Device to use. Can be 'cpu', 'mps', or None (auto-detect).
                   Also accepts torch.device objects.
        """
        if device is None:
            cls._default_device = None
        else:
            cls._default_device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """Get the current device of the model.
        
        Returns:
            torch.device: The device the model is currently on
        """
        if self._device is None:
            if self._default_device is not None:
                self._device = self._default_device
            else:
                self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.to(self._device)
        return self._device

    @device.setter
    def device(self, device: Union[torch.device, str]) -> None:
        """Set the device for this model instance.
        
        Args:
            device: Device to use. Can be 'cpu', 'mps', or a torch.device object.
        """
        new_device = torch.device(device)
        if self._device != new_device:
            self._device = new_device
            self.to(self._device)

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
        self.load_state_dict(torch.load(file_name, map_location=self.device))
