import torch
import time
import argparse
from typing import Dict
from tabulate import tabulate
from game import SnakeGameAI
from model_dqn import Linear_QNet
from model_pg import PolicyNet
from agent import Agent

def run_benchmark(model, device: str, duration: int = 60) -> Dict:
    """Run benchmark for specified duration and return metrics."""
    model.to(device)
    game = SnakeGameAI()
    agent = Agent()  # Create agent for state calculation
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
            else:  # PolicyNet
                state_old_tensor = torch.tensor(state_old, dtype=torch.float).to(device)
                action, _ = model.get_action(state_old_tensor)
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

def main():
    parser = argparse.ArgumentParser(description='Run Snake AI benchmarks')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration in seconds to run each benchmark (default: 60)')
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    # Model parameters
    input_size = 11
    hidden_size = 256
    output_size = 3

    # Initialize models
    dqn_model = Linear_QNet(input_size, hidden_size, output_size)
    pg_model = PolicyNet(input_size, hidden_size, output_size)
    
    # Store all results
    results = []
    
    for device in devices:
        print(f"\nRunning benchmarks on {device}...")
        
        # Try to load trained models if available
        try:
            dqn_model.load_state_dict(torch.load('./model/model.pth', map_location=device, weights_only=True))
            print("Loaded DQN model successfully")
        except FileNotFoundError:
            print("No trained DQN model found, using untrained model")
        except Exception as e:
            print(f"Error loading DQN model: {e}, using untrained model")
            
        try:
            pg_model.load_state_dict(torch.load('./model/model_pg.pth', map_location=device, weights_only=True))
            print("Loaded PG model successfully")
        except FileNotFoundError:
            print("No trained PG model found, using untrained model")
        except Exception as e:
            print(f"Error loading PG model: {e}, using untrained model")
        
        print("\nTesting DQN model...")
        dqn_metrics = run_benchmark(dqn_model, device, args.duration)
        results.append({
            'Model': 'DQN',
            'Device': device,
            'Games': dqn_metrics['total_games'],
            'Avg Score': f"{dqn_metrics['avg_score']:.1f}",
            'Max Score': dqn_metrics['max_score'],
            'Moves/sec': f"{dqn_metrics['moves_per_second']:.1f}"
        })
        
        print("\nTesting PG model...")
        pg_metrics = run_benchmark(pg_model, device, args.duration)
        results.append({
            'Model': 'PG',
            'Device': device,
            'Games': pg_metrics['total_games'],
            'Avg Score': f"{pg_metrics['avg_score']:.1f}",
            'Max Score': pg_metrics['max_score'],
            'Moves/sec': f"{pg_metrics['moves_per_second']:.1f}"
        })
    
    # Print consolidated results table
    print("\nBenchmark Results:")
    print(tabulate(results, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    main()
