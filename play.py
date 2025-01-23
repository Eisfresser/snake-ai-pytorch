import argparse
import torch
from game import SnakeGameAI
from model_dqn import Linear_QNet
from model_ppo import ActorCritic
import pygame

from agent import Agent

def get_model(model_type, model_path):
    """Load the specified model type with weights from model_path."""
    # Model parameters (these should match training)
    input_size = 11
    hidden_size = 256
    output_size = 3  # [straight, right turn, left turn]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if model_type == "dqn":
        model = Linear_QNet(input_size, hidden_size, output_size, device)
    elif model_type == "ppo":
        model = ActorCritic(input_size, hidden_size, output_size, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(model_path)
    return model

def play(model_type, model_path):
    """Play the Snake game using the specified model."""
    # Initialize game and model
    game = SnakeGameAI()
    model = get_model(model_type, model_path)
    model.eval()  # Set to evaluation mode
    SPEED = 25

    # Game loop
    while True:
        # Get current state
        state = Agent.get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float).to(model.device)

        # Get move based on model type
        with torch.no_grad():
            if model_type == "dqn":
                prediction = model(state_tensor)
                final_move = torch.argmax(prediction).item()
            elif model_type == "ppo":
                action, _, _ = model.get_action(state_tensor)
                final_move = action
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Play step
        reward, game_over, score = game.play_step(final_move)

        # Update display
        game._update_ui()
        pygame.display.flip()
        game.clock.tick(SPEED)

        if game_over:
            print(f"Game Over! Final Score: {score}")
            break

def main():
    parser = argparse.ArgumentParser(description='Play Snake using a trained model')
    parser.add_argument('model_type', choices=['dqn', 'ppo'], help='Type of model to use')
    parser.add_argument('--model_path', help='Path to model weights', 
                      default=None)
    args = parser.parse_args()

    # If model_path not specified, use default based on model type
    if args.model_path is None:
        args.model_path = f"model_{args.model_type}.pth"

    try:
        print(f'\n\nModel path {args.model_path} loaded successfully')
        play(args.model_type, args.model_path)
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
