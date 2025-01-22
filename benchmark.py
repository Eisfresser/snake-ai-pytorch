import torch
import time
import argparse
from typing import Dict
from tabulate import tabulate
from game import SnakeGameAI
from model_dqn import Linear_QNet, QTrainer
from model_ppo import ActorCritic, PPOTrainer
from agent import Agent
import psutil
import numpy as np

def run_benchmark(model, device: str, duration: int = 60) -> Dict:
    """Run benchmark for specified duration and return metrics."""
    model.to(device)
    game = SnakeGameAI()
    agent = Agent(device)  # Create agent for state calculation
    metrics = {
        'total_games': 0,
        'total_score': 0,
        'total_moves': 0,
        'max_score': 0
    }
    
    start_time = time.time()
    while time.time() - start_time < duration:
        game.reset()
        done = False
        
        while not done and time.time() - start_time < duration:
            state_old = agent.get_state(game)
            
            # Get move based on model type
            if isinstance(model, Linear_QNet):
                state_old_tensor = torch.tensor(state_old, dtype=torch.float).to(device)
                prediction = model(state_old_tensor)
                final_move = [0, 0, 0]
                final_move[torch.argmax(prediction).item()] = 1
            else:  # ActorCritic
                state_old_tensor = torch.tensor(state_old, dtype=torch.float).to(device)
                action, _, _ = model.get_action(state_old_tensor)  # Unpack all three values, ignore log_prob and value
                final_move = [0, 0, 0]
                final_move[action] = 1
            
            reward, done, score = game.play_step(final_move)
            metrics['total_moves'] += 1
        
        metrics['total_games'] += 1
        metrics['total_score'] += score
        metrics['max_score'] = max(metrics['max_score'], score)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate final metrics
    metrics['avg_score'] = metrics['total_score'] / metrics['total_games']
    metrics['moves_per_second'] = metrics['total_moves'] / elapsed_time
    
    return metrics

def run_training_benchmark(model, device: str, duration: int = 60) -> Dict:
    """Run training benchmark for specified duration and return metrics."""
    model.to(device)
    game = SnakeGameAI()
    agent = Agent(device)
    process = psutil.Process()
    
    # Create trainers and ensure they use the same device as the model
    if isinstance(model, Linear_QNet):
        trainer = QTrainer(model, lr=0.001, gamma=0.9)
        trainer.model.to(device)  # Ensure model in trainer is on correct device
    else:  # ActorCritic
        trainer = PPOTrainer(model, lr=0.001, gamma=0.9)
        trainer.model.to(device)  # Ensure model in trainer is on correct device
    
    metrics = {
        'total_training_steps': 0,
        'total_episodes': 0,
        'avg_memory_usage_mb': 0,
        'peak_memory_usage_mb': 0,
        'avg_loss': 0
    }
    
    memory_samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get sample training data
        state = agent.get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)

        # Get action based on model type
        if isinstance(model, ActorCritic):
            action, log_prob, _ = model.get_action(state_tensor)  # For PPO
            final_move = [0, 0, 0]
            final_move[action] = 1
        else:  # DQN
            prediction = model(state_tensor)
            final_move = [0, 0, 0]
            final_move[torch.argmax(prediction).item()] = 1

        reward = torch.tensor([0], dtype=torch.float).to(device)  # Convert reward to tensor on device
        next_state = state  # Use same state for simplicity
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).to(device)
        done = False
        
        # Record metrics
        metrics['total_training_steps'] += 1
        
        # Perform training step
        if isinstance(trainer, QTrainer):
            trainer.train_step(state_tensor, torch.tensor(final_move, dtype=torch.long).to(device), reward, next_state_tensor, done)
        else:  # PPOTrainer
            # Store experience in PPO memory buffers
            trainer.states.append(state)
            trainer.actions.append(action)
            trainer.rewards.append(reward.item())
            trainer.dones.append(done)
            trainer.log_probs.append(log_prob)
            
            # Train every 10 steps for PPO
            if len(trainer.states) >= 10:
                trainer.train_step(state_tensor, action, reward, next_state_tensor, done)
                # Clear memory after training
                trainer.states.clear()
                trainer.actions.clear()
                trainer.rewards.clear()
                trainer.dones.clear()
                trainer.log_probs.clear()
                trainer.values.clear()
                metrics['total_episodes'] += 1
        
        # Track memory usage
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_samples.append(memory_mb)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate final metrics
    metrics['training_steps_per_second'] = metrics['total_training_steps'] / elapsed_time
    metrics['avg_memory_usage_mb'] = np.mean(memory_samples)
    metrics['peak_memory_usage_mb'] = np.max(memory_samples)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Run Snake AI benchmarks')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration in seconds to run each benchmark (default: 60)')
    parser.add_argument('--type', type=str, choices=['play', 'train', 'both'], 
                      default='both', help='Type of benchmark to run')
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    # Model parameters
    input_size = 11
    hidden_size = 256
    output_size = 3

    # Store all results
    results = []
    
    for device in devices:
        print(f"\nRunning benchmarks on {device}...")
        
        # Initialize models for this device
        dqn_model = Linear_QNet(input_size, hidden_size, output_size, device)
        dqn_model.to(device)
        ppo_model = ActorCritic(input_size, hidden_size, output_size, device)
        ppo_model.to(device)
        
        # Try to load trained models if available
        try:
            state_dict = torch.load('./model/model.pth', map_location=device, weights_only=False)
            dqn_model.load_state_dict(state_dict)
            print("Loaded DQN model successfully")
        except FileNotFoundError:
            print("No trained DQN model found, using untrained model")
        except Exception as e:
            print(f"Error loading DQN model: {e}, using untrained model")
            
        try:
            state_dict = torch.load('./model/model_ppo.pth', map_location=device, weights_only=False)
            ppo_model.load_state_dict(state_dict)
            print("Loaded PPO model successfully")
        except FileNotFoundError:
            print("No trained PPO model found, using untrained model")
        except Exception as e:
            print(f"Error loading PPO model: {e}, using untrained model")

        if args.type in ['play', 'both']:
            print("\nTesting DQN model gameplay...")
            dqn_play_metrics = run_benchmark(dqn_model, device, args.duration)
            results.append({
                'Model': 'DQN',
                'Type': 'Play',
                'Device': device,
                'Games': dqn_play_metrics['total_games'],
                'Avg Score': round(dqn_play_metrics['avg_score'], 2),
                'Max Score': dqn_play_metrics['max_score'],
                'Moves/sec': round(dqn_play_metrics['moves_per_second'], 2)
            })
            
            print("\nTesting PPO model gameplay...")
            ppo_play_metrics = run_benchmark(ppo_model, device, args.duration)
            results.append({
                'Model': 'PPO',
                'Type': 'Play',
                'Device': device,
                'Games': ppo_play_metrics['total_games'],
                'Avg Score': round(ppo_play_metrics['avg_score'], 2),
                'Max Score': ppo_play_metrics['max_score'],
                'Moves/sec': round(ppo_play_metrics['moves_per_second'], 2)
            })

        if args.type in ['train', 'both']:
            print("\nTesting DQN model training...")
            dqn_train_metrics = run_training_benchmark(dqn_model, device, args.duration)
            results.append({
                'Model': 'DQN',
                'Type': 'Train',
                'Device': device,
                'Training Steps': dqn_train_metrics['total_training_steps'],
                'Episodes': dqn_train_metrics['total_episodes'],
                'Steps/sec': round(dqn_train_metrics['training_steps_per_second'], 2),
                'Avg Memory (MB)': round(dqn_train_metrics['avg_memory_usage_mb'], 2),
                'Peak Memory (MB)': round(dqn_train_metrics['peak_memory_usage_mb'], 2)
            })
            
            print("\nTesting PPO model training...")
            ppo_train_metrics = run_training_benchmark(ppo_model, device, args.duration)
            results.append({
                'Model': 'PPO',
                'Type': 'Train',
                'Device': device,
                'Training Steps': ppo_train_metrics['total_training_steps'],
                'Episodes': ppo_train_metrics['total_episodes'],
                'Steps/sec': round(ppo_train_metrics['training_steps_per_second'], 2),
                'Avg Memory (MB)': round(ppo_train_metrics['avg_memory_usage_mb'], 2),
                'Peak Memory (MB)': round(ppo_train_metrics['peak_memory_usage_mb'], 2)
            })
        
    # Print results in a nice table
    play_results = [r for r in results if r['Type'] == 'Play']
    train_results = [r for r in results if r['Type'] == 'Train']
    
    if play_results:
        play_headers = ['Model', 'Type', 'Device', 'Games', 'Avg Score', 'Max Score', 'Moves/sec']
        play_table = [[result[col] for col in play_headers] for result in play_results]
        print("\nGameplay Benchmark Results:")
        print(tabulate(play_table, headers=play_headers, tablefmt="grid"))
        
    if train_results:
        train_headers = ['Model', 'Type', 'Device', 'Training Steps', 'Episodes', 'Steps/sec', 'Avg Memory (MB)', 'Peak Memory (MB)']
        train_table = [[result[col] for col in train_headers] for result in train_results]
        print("\nTraining Benchmark Results:")
        print(tabulate(train_table, headers=train_headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
