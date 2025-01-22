import torch
import time
from typing import Dict, List
from tabulate import tabulate
from collections import defaultdict
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
                action, _ = model.get_action(state_old)
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
    # Model parameters
    input_size = 11
    hidden_size = 256
    output_size = 3
    
    # Initialize models
    dqn_model = Linear_QNet(input_size, hidden_size, output_size)
    pg_model = PolicyNet(input_size, hidden_size, output_size)
    
    # Try to load trained models if available
    try:
        dqn_model.load_state_dict(torch.load('./model/model.pth'))
        print("Loaded DQN model successfully")
    except:
        print("No trained DQN model found, using untrained model")
        
    try:
        pg_model.load_state_dict(torch.load('./model/model_pg.pth'))
        print("Loaded PG model successfully")
    except:
        print("No trained PG model found, using untrained model")
    
    # Devices to test
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    # Run benchmarks
    results = []
    for device in devices:
        print(f"\nRunning benchmarks on {device}...")
        
        # DQN benchmark
        print("Testing DQN model...")
        dqn_metrics = run_benchmark(dqn_model, device)
        results.append({
            'Model': 'DQN',
            'Device': device,
            'Games': dqn_metrics['total_games'],
            'Avg Score': f"{dqn_metrics['avg_score']:.1f}",
            'Max Score': dqn_metrics['max_score'],
            'Moves/sec': f"{dqn_metrics['moves_per_second']:.1f}"
        })
        
        # PG benchmark
        print("Testing PG model...")
        pg_metrics = run_benchmark(pg_model, device)
        results.append({
            'Model': 'PG',
            'Device': device,
            'Games': pg_metrics['total_games'],
            'Avg Score': f"{pg_metrics['avg_score']:.1f}",
            'Max Score': pg_metrics['max_score'],
            'Moves/sec': f"{pg_metrics['moves_per_second']:.1f}"
        })
    
    # Print results table
    print("\nBenchmark Results:")
    print(tabulate(results, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    main()
