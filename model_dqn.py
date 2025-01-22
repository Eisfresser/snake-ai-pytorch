import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import Tuple, List, Union
from model_base import BaseModel
import warnings
# Suppress tensor construction warnings
warnings.filterwarnings('ignore', message='To copy construct from a tensor*')

class Linear_QNet(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 device: str) -> None:
        super().__init__(device=device)
        self._init_layers(input_size, hidden_size, output_size)

    def _init_layers(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.linear2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model: Linear_QNet, lr: float, gamma: float) -> None:
        self.lr: float = lr
        self.gamma: float = gamma
        self.model: Linear_QNet = model
        self.device = self.model.device
        self.optimizer: optim.Adam = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion: nn.MSELoss = nn.MSELoss().to(self.device)
        self.model.to(self.device)  # Ensure model is on correct device

    def train_step(self, state: List[float], action: List[int], reward: float, 
                  next_state: List[float], done: Union[bool, Tuple[bool, ...]]) -> None:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Get current Q values
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_state_idx = next_state[idx].unsqueeze(0)  # Add batch dimension
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_idx))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
