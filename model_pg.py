import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.policy(x)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def save(self, file_name='model_pg.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PGTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # Store episode data
        self.reset_episode()
    
    def reset_episode(self):
        self.log_probs = []
        self.rewards = []
        
    def remember(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensor just to maintain interface compatibility
        state = torch.tensor(state, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
        
        # Get action and log probability
        _, log_prob = self.model.get_action(state)
        
        # Store the transition
        self.remember(log_prob, reward)
        
        # Only update policy at the end of episode
        if done:
            # Calculate discounted rewards
            returns = []
            G = 0
            for r in reversed(self.rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, R in zip(self.log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Reset episode data
            self.reset_episode()
