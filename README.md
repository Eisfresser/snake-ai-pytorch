# Snake for CAS AML M6 RL Uni Bern

Based on [snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch) by [Patrick Loeber](https://github.com/patrickloeber)

```bash
# Install dependencies
uv sync

# Train a model, model type is dqn/ppo, best model trained goes to ./model
uv run train.py --model dqn

# Play a game, models are dqn/ppo
uv run play.py --model dqn

# Benchmark a model, type is play/train/both, duration for each test in seconds
uv run benchmark.py --type both --duration 45
```

