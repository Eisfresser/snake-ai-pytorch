import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from typing import List, Tuple, Union, Dict
from model_base import BaseModel

class ActorCritic(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str) -> None:
        super().__init__(device=device)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For compatibility with DQN interface, return only actor output
        return self.actor(x)
    
    def forward_actor_critic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # For PPO training, return both actor and critic outputs
        return self.actor(x), self.critic(x)

    def get_action(self, state: Union[List[float], torch.Tensor]) -> Tuple[Union[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        probs, value = self.forward_actor_critic(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # If single state, return scalar action
        if len(state) == 1:
            return action.item(), log_prob, value
        # If batch of states, return tensor of actions
        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, values = self.forward_actor_critic(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def save(self, file_name: str = 'model_ppo.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PPOTrainer:
    def __init__(self, 
                 model: ActorCritic, 
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 epochs: int = 10,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 batch_size: int = 32) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        self.epochs = epochs    # Number of epochs to update policy
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        
        # Memory buffers for experience collection
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def _to_tensor(self, x, dtype=torch.float):
        """Convert input to tensor with batch dimension"""
        if isinstance(x, (int, float)):
            x = torch.tensor([x], dtype=dtype, device=self.model.device)
        elif isinstance(x, (tuple, list)):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(dtype).to(self.model.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        return x

    def train_step(self, state, action, reward, next_state, done):
        # Convert all inputs to batched tensors
        state = self._to_tensor(state)
        action = self._to_tensor(action)
        reward = self._to_tensor(reward).view(-1, 1)  # Shape: [batch_size, 1]
        next_state = self._to_tensor(next_state)
        done = self._to_tensor(done).view(-1, 1)  # Shape: [batch_size, 1]

        # Get current action probabilities and value
        with torch.no_grad():
            _, log_prob, value = self.model.get_action(state)
            next_value = self.model.forward_actor_critic(next_state)[1]

        # Store experience in memory (keeping batch dimension)
        self.states.append(state)  # Shape: [batch_size, state_dim]
        self.actions.append(torch.argmax(action, dim=-1))  # Shape: [batch_size]
        self.rewards.append(reward)  # Shape: [batch_size, 1]
        self.dones.append(done)  # Shape: [batch_size, 1]
        self.log_probs.append(log_prob)  # Shape: [batch_size]
        self.values.append(value)  # Shape: [batch_size, 1]

        # Only update when we have enough samples
        if len(self.states) >= self.batch_size:
            # Concatenate all batched experiences
            states = torch.cat(self.states, dim=0)  # Shape: [total_samples, state_dim]
            actions = torch.cat(self.actions, dim=0)  # Shape: [total_samples]
            old_log_probs = torch.cat(self.log_probs, dim=0)  # Shape: [total_samples]
            values = torch.cat(self.values, dim=0)  # Shape: [total_samples, 1]
            rewards = torch.cat(self.rewards, dim=0)  # Shape: [total_samples, 1]
            dones = torch.cat(self.dones, dim=0)  # Shape: [total_samples, 1]
            
            # For batched data, we'll use the last value as next_value for each sequence
            # Assuming the sequences are independent
            last_value = next_value[-1] if next_value.dim() > 0 else next_value

            # Calculate returns and advantages
            returns, advantages = self.compute_gae(
                rewards.squeeze().tolist(), 
                values.squeeze().tolist(),
                last_value.item(),
                dones.squeeze().tolist()
            )
            
            # Convert returns and advantages to tensors
            returns = torch.FloatTensor(returns).to(self.model.device)
            advantages = torch.FloatTensor(advantages).to(self.model.device)
            
            # Perform PPO update
            self.update(states, actions, old_log_probs, returns, advantages)
            
            # Clear memory buffers
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.dones.clear()
            self.log_probs.clear()
            self.values.clear()

    def compute_gae(self, 
                    rewards: List[float], 
                    values: List[float], 
                    next_value: float,
                    dones: List[bool],
                    gamma: float = 0.99,
                    lambda_: float = 0.95) -> Tuple[List[float], List[float]]:
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
                
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            gae = delta + gamma * lambda_ * next_non_terminal * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return returns, advantages

    def update(self, 
               states: torch.Tensor,
               actions: torch.Tensor,
               old_log_probs: torch.Tensor,
               returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict[str, float]:
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for specified number of epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            # Get new log probs, values and entropy
            new_log_probs, values, entropy = self.model.evaluate_actions(states, actions)
            
            # Calculate ratio between new and old probabilities
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Calculate policy loss using PPO clipped objective
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Calculate entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.value_coef * value_loss + 
                   self.entropy_coef * entropy_loss)
            
            # Perform update
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosive gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()

        num_updates = self.epochs
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
