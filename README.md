🚀 LLMs From Scratch — Base Model → SFT → Reward Model → PPO & GRPO RLHF
A Complete, End-to-End Implementation of Modern LLM Training & Alignment Using Pure PyTorch
📘 Overview

This repository provides a full-stack implementation of modern LLM development, entirely using pure PyTorch, without relying on high-level training frameworks.
It walks through the complete lifecycle of building, training, scaling, and aligning a Large Language Model:

Foundations & core architecture

Tiny LLM training loop from scratch

Modern architectural upgrades (RoPE, RMSNorm, KV-cache, etc.)

Scaling techniques for larger models

Mixture-of-Experts (MoE)

Supervised Fine-Tuning (SFT)

Reward Model training

RLHF via PPO

RLHF via GRPO (Group-Relative Policy Optimization)

This repository is intended for ML Engineers, AI Researchers, and LLM Practitioners looking for a transparent, hands-on, production-grade understanding of every component behind modern LLMs.

⭐ Contents
Part 0 — Foundations & Mindset
Part 1 — Core Transformer Architecture
Part 2 — Training a Tiny LLM
Part 3 — Modernizing the Architecture
Part 4 — Scaling Up
Part 5 — Mixture-of-Experts (MoE)
Part 6 — Supervised Fine-Tuning (SFT)
Part 7 — Reward Modeling
Part 8 — RLHF with PPO
Part 9 — RLHF with GRPO


Each part includes conceptual explanations, PyTorch implementations, and runnable training/inference scripts.

🧩 Part 0 — Foundations & Mindset
0.1 High-Level LLM Training Pipeline

Pretraining → SFT → RM → RLHF

Why alignment matters

Differences between base models vs aligned models

0.2 Environment Setup
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt


Includes GPU setup, CUDA, mixed precision, and profiling tools.

🧠 Part 1 — Core Transformer Architecture

A full GPT-style decoder implemented step-by-step:

Positional embeddings (learned + sinusoidal)

Self-attention from first principles (manual toy example)

Single attention head implementation

Multi-head attention (splitting, projections, merging)

Feed-forward MLP layers (GELU, expansion)

Residual connections & LayerNorm (pre-LN)

Stacking multiple blocks into a full transformer

Causal masking & autoregressive forward pass

This part builds a minimal, fully working GPT.

🏋️‍♂️ Part 2 — Training a Tiny LLM

You train a small GPT-like model from scratch:

Byte-level tokenizer

Dataset batching & label shifting

Cross-entropy objective for next-token prediction

Full training loop (optimizer, scheduler, logging)

Sampling strategies:

temperature

top-k

top-p (nucleus sampling)

Includes evaluation on validation split and generation examples.

🔧 Part 3 — Modernizing the Architecture

Modern components used in GPT-NeoX, LLaMA, Mistral, etc.:

RMSNorm vs LayerNorm

RoPE (Rotary Positional Embeddings)

SwiGLU activations (used in LLaMA/Mistral)

KV cache for fast autoregressive inference

Sliding-window attention

Attention sink trick

Rolling KV buffer for streaming inference

This upgrades the tiny model → practically usable modern architecture.

📈 Part 4 — Scaling Up

Techniques enabling larger models:

BPE tokenization (switch from byte-level)

Gradient accumulation

Mixed precision (fp16/bf16 via AMP)

Learning rate schedules & warmup

Long training runs: checkpointing & resuming

TensorBoard / Weights & Biases logging

🧮 Part 5 — Mixture-of-Experts (MoE)

Advanced architecture used in Mixtral, Switch-Transformer, DeepSeek-R1:

MoE theory: sparse routing, gating networks

Load balancing losses

PyTorch implementation of MoE layers

Hybrid dense + expert blocks

This part lays the foundation for training efficient large sparse models.

🎓 Part 6 — Supervised Fine-Tuning (SFT)

Training the model on instruction-following data:

Prompt + response formatting

Causal LM loss with masked labels

Curriculum learning for instructions

Evaluation against reference outputs

This yields an SFT policy, serving as the base for alignment.

🏆 Part 7 — Reward Modeling

Building a model that scores completions according to human preferences:

Pairwise ranking datasets ("chosen vs rejected")

Reward model architecture: transformer + scalar head

Bradley–Terry loss

Margin ranking loss

Reward sanity checks / calibration

This produces the reward signal for RLHF.

🤖 Part 8 — RLHF with PPO

Full RLHF pipeline with Proximal Policy Optimization, as used in InstructGPT:

8.1 Policy Network

SFT model + value head for reward prediction.

8.2 Reward Signal

Reward model from Part 7 + KL penalty vs reference model.

8.3 PPO Objective

Balancing reward maximization with policy stability.

8.4 RLHF Training Loop

Sample prompts

Generate completions

Compute rewards

Update policy using PPO

Normalize advantages

KL-controlled rollouts

8.5 Stabilization Techniques

Reward normalization

Adaptive KL penalty

Gradient clipping

Early stopping on KL explosion

This part yields an aligned policy model.

🔥 Part 9 — RLHF with GRPO

A modern, simpler, value-head-free RLHF algorithm (used by DeepSeek-R1, Qwen2.5, open-source RLHF systems):

Why GRPO?

No value function

More stable

Lower variance

Simpler implementation

Better suited to small/medium-scale training

GRPO Pipeline

Sample k completions per prompt

Compute reward for each

Compute group mean reward

Advantage = reward − group mean

PPO-like clipped objective

Explicit KL(π‖π_ref) penalty

This part yields a production-ready, value-free RLHF implementation.

🧪 Evaluation & Monitoring

Throughout the pipeline, notebooks track:

Loss curves (pretraining, SFT, PPO, GRPO)

KL divergence

Reward score distribution

Group-relative reward variance

Side-by-side model generations:

Base model

SFT model

PPO-aligned model

GRPO-aligned model

⚙️ Installation
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt

🏗️ Repository Structure
LLMs_from_scratch/
│
├── part_0/   # Foundations & setup
├── part_1/   # Transformer architecture
├── part_2/   # Tiny GPT training loop
├── part_3/   # Modern components (RoPE, RMSNorm, KV cache)
├── part_4/   # Scaling techniques
├── part_5/   # Mixture-of-Experts
├── part_6/   # Supervised Fine-Tuning (SFT)
├── part_7/   # Reward Modeling
├── part_8/   # PPO RLHF
├── part_9/   # GRPO RLHF
├── utils/    # tokenizers, dataloaders, logging, configs
└── README.md

🧱 Tech Stack

Python 3.11

PyTorch

CUDA / cuDNN

WandB / TensorBoard

BPE & byte-level tokenizers

📄 License

MIT License — free to use, modify, extend.

🎯 Who Is This Project For?

This repository is ideal for:

ML/AI Engineers wanting to truly understand LLM internals

Researchers testing new alignment algorithms (DPO, ORPO, RLAIF…)

Teams training internal LLMs

Anyone wanting to reproduce the full LLM → SFT → RLHF pipeline without black-box libraries
