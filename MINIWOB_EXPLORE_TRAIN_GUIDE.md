# MiniWob++ Exploration and Training Guide

This guide explains how to use the MiniWob++ exploration and training capabilities in Go-Browse.

## Overview

The MiniWob++ adaptation includes:
- **Exploration**: Collect training data by running agents on MiniWob++ tasks
- **Training**: Train models using supervised fine-tuning on collected trajectories
- **Evaluation**: Assess trained models on MiniWob++ benchmarks

## Quick Start

### 1. Setup Environment

```bash
# Install MiniWob++ (if not already installed)
pip install miniwob

# Set up Chrome environment
python setup_chrome.py

# Check that everything is working
python run_miniwob_exploration.py setup
```

### 2. Run Exploration

```bash
# List available MiniWob++ environments
python run_miniwob_exploration.py list

# Run exploration with basic configuration
python run_miniwob_exploration.py run --config configs/miniwob_explore_config.yaml

# Run exploration with advanced configuration (more environments)
python run_miniwob_exploration.py run --config configs/miniwob_explore_advanced.yaml
```

### 3. Analyze Collected Data

```bash
# Analyze exploration results
python run_miniwob_training.py analyze ./exploration_results/miniwob_basic
```

### 4. Train Model

```bash
# Check training requirements
python run_miniwob_training.py check

# Train model on collected data
python run_miniwob_training.py train --config configs/miniwob_train_config.yaml
```

## Detailed Usage

### Exploration

#### Configuration Files

**Basic Configuration** (`configs/miniwob_explore_config.yaml`):
- 21 basic MiniWob++ environments
- 10 episodes per environment
- Azure GPT-4o agent
- Saves to `./exploration_results/miniwob_basic`

**Advanced Configuration** (`configs/miniwob_explore_advanced.yaml`):
- 40+ MiniWob++ environments including complex tasks
- 20 episodes per environment
- Bedrock Claude agent for exploration
- Azure GPT-4o agent for evaluation
- Saves to `./exploration_results/miniwob_advanced`

#### Custom Exploration

Create a custom configuration:

```bash
python run_miniwob_exploration.py create-config \
    --output my_config.yaml \
    --env_names click-test click-button text-input \
    --episodes_per_env 15 \
    --agent_type azure
```

Run with custom parameters:

```bash
python run_miniwob_exploration.py run \
    --config configs/miniwob_explore_config.yaml \
    --exp_dir ./my_exploration \
    --episodes_per_env 5 \
    --env_names click-test click-button
```

#### Exploration Output Structure

```
exploration_results/
├── exploration_summary.json          # Overall statistics
├── explore_config.yaml               # Configuration used
├── click-test/                       # Environment directory
│   ├── task_info.json               # Task metadata
│   ├── positive_trajs/              # Successful trajectories
│   │   ├── 0/                       # Trajectory 0
│   │   │   ├── trajectory.json      # Trajectory data
│   │   │   └── screenshots/         # Step screenshots
│   │   └── 1/                       # Trajectory 1
│   ├── negative_trajs/              # Failed trajectories
│   └── episodes/                    # Raw episode data
└── click-button/                     # Another environment
    └── ...
```

### Training

#### Prerequisites

Install training dependencies:

```bash
pip install torch transformers datasets trl wandb
```

Set up Weights & Biases (optional):

```bash
export WANDB_API_KEY="your_wandb_api_key"
```

#### Training Configuration

The training configuration (`configs/miniwob_train_config.yaml`) includes:

- **Model**: Base model to fine-tune (default: Qwen2.5-7B-Instruct)
- **Data**: Exploration directory and filtering options
- **Training**: Hyperparameters and optimization settings
- **Logging**: Weights & Biases integration

#### Running Training

```bash
# Basic training
python run_miniwob_training.py train --config configs/miniwob_train_config.yaml

# Override specific parameters
python run_miniwob_training.py train \
    --config configs/miniwob_train_config.yaml \
    --exploration_dir ./exploration_results/miniwob_advanced \
    --output_dir ./models/my_miniwob_model \
    --epochs 5 \
    --batch_size 2
```

#### Training Output

```
models/miniwob_sft/
├── config.json                      # Model configuration
├── pytorch_model.bin                # Trained model weights
├── tokenizer.json                   # Tokenizer
├── training_dataset.jsonl           # Prepared training data
├── training_dataset_stats.json      # Dataset statistics
└── training_logs/                   # Training logs and checkpoints
```

## Advanced Features

### Multi-Agent Exploration

Use different agents for exploration and evaluation:

```yaml
# In exploration config
explorer_agent:
  agent_factory_args:
    agent_name: "BedrockClaudeMiniWobAgent"
    model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0"
    temperature: 0.2

evaluator_agent:
  agent_factory_args:
    agent_name: "AzureGPTMiniWobAgent"
    model_name: "gpt-4o"
    temperature: 0.0
```

### Data Filtering

Control which trajectories to include in training:

```yaml
# In training config
include_negative: false               # Exclude failed trajectories
min_success_rate: 0.1                # Only include envs with >10% success
```

### Custom Data Processing

The training pipeline supports custom data processing:

```python
from webexp.train.miniwob_sft_policy import MiniWobSFTTrainer

trainer = MiniWobSFTTrainer(model_id="your-model")

# Custom dataset preparation
trainer.prepare_dataset_from_exploration(
    exploration_dir="./exploration_results",
    output_path="./custom_dataset.jsonl",
    include_negative=True,
    min_success_rate=0.05
)
```

## Environment-Specific Notes

### Supported MiniWob++ Environments

The framework supports all MiniWob++ environments, including:

**Basic Interaction**:
- `click-test`, `click-button`, `click-checkboxes`
- `text-input`, `focus-text`, `enter-text`

**Complex Tasks**:
- `book-flight`, `login-user`, `email-inbox`
- `social-media`, `search-engine`

**Mathematical**:
- `simple-algebra`, `simple-arithmetic`
- `guess-number`, `count-shape`

**Advanced Interaction**:
- `drag-box`, `use-autocomplete`, `use-slider`
- `scroll-text`, `resize-textarea`

### Environment-Specific Tips

1. **Text Input Tasks**: Ensure proper text formatting in action processing
2. **Drag Tasks**: May require multiple attempts due to coordinate precision
3. **Complex Forms**: Break down into smaller steps for better success rates
4. **Time-Sensitive Tasks**: Consider increasing `slow_mo` parameter

## Troubleshooting

### Common Issues

**Chrome/ChromeDriver Issues**:
```bash
# Run Chrome setup
python setup_chrome.py

# Or use the installation script
./install_chrome.sh
```

**MiniWob++ Import Errors**:
```bash
pip install miniwob gymnasium
```

**Training Memory Issues**:
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `fp16=True` or `bf16=True`

**Low Success Rates**:
- Increase `episodes_per_env` for more attempts
- Adjust agent `temperature` (lower for more deterministic)
- Check environment-specific requirements

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Save detailed episode data:

```yaml
# In exploration config
save_screenshots: true
save_traces: true
```

## Performance Optimization

### Exploration

- Use `headless: true` for faster execution
- Adjust `slow_mo` based on environment stability
- Run multiple environments in parallel (future feature)

### Training

- Use appropriate batch size for your GPU memory
- Enable gradient checkpointing for large models
- Use mixed precision training (fp16/bf16)

## Integration with Existing Workflows

### Using Trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model
model = AutoModelForCausalLM.from_pretrained("./models/miniwob_sft")
tokenizer = AutoTokenizer.from_pretrained("./models/miniwob_sft")

# Use in MiniWob++ agent
# (Integration with agent framework coming soon)
```

### Evaluation

```bash
# Run benchmark evaluation
python -m webexp.benchmark.run_miniwob \
    --config configs/benchmark_miniwob.yaml \
    --model_path ./models/miniwob_sft
```

## Future Enhancements

- **Reinforcement Learning**: RL training on MiniWob++ environments
- **Multi-Task Learning**: Joint training across multiple environments
- **Few-Shot Learning**: Adaptation to new environments with minimal data
- **Hierarchical Policies**: Decomposition of complex tasks
- **Interactive Training**: Human-in-the-loop data collection

## Contributing

To contribute to the MiniWob++ adaptation:

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure compatibility with both Azure AI and Bedrock agents

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the Chrome setup guide: `CHROME_SETUP_GUIDE.md`
- Check existing GitHub issues
- Create a new issue with detailed information