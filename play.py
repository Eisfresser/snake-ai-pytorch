import argparse
import torch
from game import SnakeGameAI, Point, Direction, SPEED
from model_dqn import Linear_QNet
from model_ppo import ActorCritic
from model_old_pg import PolicyNet
import pygame
from agent import Agent
import numpy as np

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
    elif model_type == "pg":
        model = PolicyNet(input_size, hidden_size, output_size, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(model_path)
    return model

def get_state(game):
    """Get the current state of the game."""
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
        
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
    ]

    return np.array(state, dtype=int)

def play(model_type, model_path):
    """Play the Snake game using the specified model."""
    # Initialize game and model
    game = SnakeGameAI()
    model = get_model(model_type, model_path)
    model.eval()  # Set to evaluation mode

    # Game loop
    while True:
        # Get current state
        state = get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float).to(model.device)

        # Get move based on model type
        with torch.no_grad():
            if model_type == "dqn":
                prediction = model(state_tensor)
                final_move = torch.argmax(prediction).item()
            elif model_type == "ppo":
                action_probs = model.actor(state_tensor)
                final_move = torch.argmax(action_probs).item()
            else:  # pg
                action_probs = model(state_tensor)
                final_move = torch.argmax(action_probs).item()

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
    parser.add_argument('model_type', choices=['dqn', 'ppo', 'pg'], help='Type of model to use')
    parser.add_argument('--model_path', help='Path to model weights', 
                      default=None)
    args = parser.parse_args()

    # If model_path not specified, use default based on model type
    if args.model_path is None:
        args.model_path = f"model_{args.model_type}.pth"

    try:
        play(args.model_type, args.model_path)
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
