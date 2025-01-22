import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List, Tuple, Union
from model_base import BaseModel

class PolicyNet(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str) -> None:
        super().__init__(device=device)
        self.policy: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def get_action(self, state: Union[List[float], torch.Tensor]) -> Tuple[Union[int, torch.Tensor], torch.Tensor]:
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # If single state, return scalar action
        if len(state) == 1:
            return action.item(), log_prob
        # If batch of states, return tensor of actions
        return action, log_prob

    def save(self, file_name: str = 'model_pg.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PGTrainer:
    def __init__(self, model: PolicyNet, lr: float, gamma: float) -> None:
        self.lr: float = lr
        self.gamma: float = gamma
        self.model: PolicyNet = model
        self.device = self.model.device
        self.optimizer: optim.Adam = optim.Adam(model.parameters(), lr=self.lr)
        
        # Store episode data
        self.reset_episode()
    
    def reset_episode(self) -> None:
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        
    def remember(self, log_prob: torch.Tensor, reward: float) -> None:
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def train_step(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        # Convert to tensor just to maintain interface compatibility
        state = torch.as_tensor(state, dtype=torch.float, device=self.device)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
        
        # Get action and log probability
        _, log_prob = self.model.get_action(state)
        if isinstance(log_prob, tuple):
            log_prob = log_prob[1]  # Extract just the log probability
        
        # Store the transition
        self.remember(log_prob, reward)
        
        # Only update policy at the end of episode
        if done:

            # self.rewards may be list of int or list of list of int, remove the outer list
            if not isinstance(self.rewards[0], int):
                self.rewards = [item for sublist in self.rewards for item in sublist]

            # Calculate discounted rewards
            returns: List[float] = []
            G: float = 0
            for r in reversed(self.rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns_tensor: torch.Tensor = torch.tensor(returns)
            
            # Normalize returns
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
            # Calculate policy loss
            policy_loss: List[torch.Tensor] = []
            for log_prob, R in zip(self.log_probs, returns_tensor):
                policy_loss.append(-log_prob * R)
            total_loss: torch.Tensor = torch.stack(policy_loss).sum()
            
            # Update policy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Reset episode data
            self.reset_episode()
