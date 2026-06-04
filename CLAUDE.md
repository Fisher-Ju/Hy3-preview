# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Hy3-preview repository, containing the 295B-parameter Mixture-of-Experts (MoE) model developed by the Tencent Hy Team. It's the first model trained on their rebuilt infrastructure and represents the strongest model they've shipped so far. The model excels in complex reasoning, instruction following, context learning, coding, and agent tasks.

## Key Components

### Model Architecture
- 295B total parameters with 21B activated parameters
- 3.8B MTP layer parameters
- 80 layers (excluding MTP layer) with 192 experts (top-8 activated)
- 64 attention heads (GQA with 8 KV heads, head dim 128)
- 4096 hidden size, 13312 intermediate size
- 256K context length
- 120832 vocabulary size
- Uses MoE (Mixture-of-Experts) architecture with shared experts

### Training Infrastructure
- Full fine-tuning and LoRA fine-tuning support
- DeepSpeed ZeRO configurations (ds_zero2_no_offload.json, ds_zero3_no_offload.json, ds_zero3_offload.json)
- LLaMA-Factory integration with custom patches
- Support for both single-machine and multi-machine training
- Specialized training scripts for different use cases

### Deployment
- vLLM and SGLang deployment options
- OpenAI-compatible API interface
- Support for speculative decoding with MTP

## Development Setup

### Training Environment
To set up the training environment:
```bash
cd train
pip install -r requirements.txt
```

### Quickstart for Training
```bash
cd train
bash train.sh
```

### Training Configuration
Key training parameters include:
- `--deepspeed`: Path to DeepSpeed config (ds_zero2_no_offload.json, ds_zero3_no_offload.json, ds_zero3_offload.json)
- `--model_name_or_path`: Path to Hy3 preview HF pre-trained model weights
- `--tokenizer_name_or_path`: Path to tokenizer folder
- `--train_data_file`: Path to training jsonl file
- `--output_dir`: Output directory for logs and model weights
- `--use_lora`: Enable LoRA training
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--learning_rate`: Maximum learning rate (default 1e-5 for full finetuning, 2e-4 for LoRA)
- `--max_steps`: Total number of training steps
- `--save_steps`: Number of steps between saving checkpoints

### Model Conversion Tools
- `train/tools/convert_ckpt_to_outer.py`: Convert checkpoint to HuggingFace-compatible format
- `train/tools/check_converted.py`: Validate converted checkpoint

### LoRA Weight Merging
- `train/merge_lora_weight.sh`: Merge LoRA weights with base model

## Common Commands

### Training Commands
- `cd train && bash train.sh` - Run single-machine training with full fine-tuning
- `cd train && bash train_lora.sh` - Run LoRA training  
- `cd train/llama_factory_support && bash train_lf.sh` - Run LLaMA-Factory training
- `cd train && python train.py --help` - Show all training arguments

### Model Conversion
- `python train/tools/convert_ckpt_to_outer.py --input_dir <path> --output_dir <path>` - Convert checkpoint format
- `python train/tools/check_converted.py <converted_checkpoint_dir>` - Validate converted checkpoint

### Deployment
- vLLM: `vllm serve tencent/Hy3-preview --tensor-parallel-size 8 --speculative-config.method mtp ...`
- SGLang: `python3 -m sglang.launch_server ...`

## Architecture Notes

### Training Pipeline
The training process uses HuggingFace Transformers and DeepSpeed for distributed training across multiple GPUs. The repository supports both full fine-tuning and LoRA fine-tuning methods with various DeepSpeed configurations. 

Key features:
- Custom patches for DeepSpeed ZeRO-3 compatibility
- Support for gradient checkpointing
- Special handling for MoE parameters and expert weights
- Integration with LLaMA-Factory for flexible training workflows

### Model Structure
The model uses a MoE (Mixture-of-Experts) architecture with:
- 192 experts per layer
- Top-8 expert activation 
- Shared experts for better generalization
- MTP (Mixed-Training-Parameter) layer for enhanced training efficiency
- Support for mixed MoE with shared experts

### Key Directories
- `/train` - Training scripts, configurations, and tools
- `/train/llama_factory_support` - LLaMA-Factory integration files
- `/train/tools` - Utility scripts for checkpoint conversion and validation
- `/assets` - Benchmark images and logos
- `/model` - Model weights and configuration files (including tokenizer, config, etc.)

## Development Guidelines

### Model Development
When working with model files:
- Pay attention to the MoE structure and expert handling
- Be mindful of the MTP layer parameters
- Consider using the provided conversion tools for checkpoint manipulation
- Handle both 3D fused experts and per-expert unfused formats correctly

### Training Development
When developing training code:
- Follow DeepSpeed configuration patterns
- Ensure compatibility with LLaMA-Factory integration
- Maintain support for both full and LoRA fine-tuning
- Apply necessary monkey-patches for DeepSpeed compatibility
- Handle gradient checkpointing properly with ZeRO-3

### Testing and Validation
- Use the provided validation scripts for checkpoint integrity
- Follow benchmark protocols for performance evaluation
- Test with various DeepSpeed configurations
- Validate training with dummy data before full runs
- Monitor training metrics and checkpoints for consistency

## Special Considerations

### Multi-node Training
For multi-node training:
1. Configure passwordless SSH access between nodes
2. Set IP_LIST environment variable with comma-separated IP addresses
3. Ensure all nodes have identical code and dependencies
4. Use proper network configuration for NCCL communications

### Performance Optimization
- Use appropriate DeepSpeed configurations based on hardware constraints
- Enable flash attention for faster training when available
- Utilize gradient checkpointing to reduce memory usage
- Adjust batch sizes and learning rates according to available resources