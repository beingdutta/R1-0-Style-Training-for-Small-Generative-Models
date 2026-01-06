# 👁️ Vision-R1-Zero: Emergent Quantitative Reasoning in VLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange)
![TRL](https://img.shields.io/badge/TRL-GRPO-green)
![License](https://img.shields.io/badge/License-MIT-purple)

> **"Reasoning should emerge from the environment, not be imitated from a dataset."**

## 🚀 Overview

**Vision-R1-Zero** explores the application of **DeepSeek R1-Zero** style training to **Vision Language Models (VLMs)**. 

Unlike traditional pipelines that rely on massive Supervised Fine-Tuning (SFT) to teach models *how* to think, this project demonstrates that small Quantized VLMs (2B & 256M parameters) can self-evolve **Chain-of-Thought (CoT)** reasoning capabilities using **only** Group Relative Policy Optimization (GRPO).

**Key Differentiator:** We skip the SFT phase entirely. The model starts with a "Cold Start" and learns to reason purely by maximizing a quantitative reward signal.

## 📊 Comprehensive Benchmarks

We conducted extensive experiments comparing **SFT (Supervised Fine-Tuning)** vs. **GRPO (Reinforcement Learning)** across multiple model sizes and benchmarks.

### 1. The "Style Drift" Phenomenon (TDIUC Dataset)
This table highlights the core finding: **GRPO improves accuracy but degrades cosine similarity** to human ground truth. This indicates the model evolves its *own* reasoning style rather than mimicking the human annotations.

| Model | Size | Strategy | Rank | **Accuracy (EM)** | **Reasoning Similarity (Cos)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL** | 2B | SFT | 32 | 81.85% | **99.42%** |
| **Qwen2-VL** | 2B | GRPO | 32 | **86.97%** 🟢 | 24.47% |
| **SmolVLM** | 500M | SFT | 32 | 70.82% | **97.38%** |
| **SmolVLM** | 500M | GRPO | 32 | **72.60%** 🟢 | 20.37% |
| **SmolVLM** | 256M | SFT | 32 | 63.70% | **94.26%** |
| **SmolVLM** | 256M | GRPO | 32 | **67.70%** 🟢 | 11.34% |

### 2. Generalization to Unseen Benchmarks (Qwen2-VL 2B)
GRPO training on the TDIUC dataset led to significant emergent generalizations on completely different counting benchmarks (TallyQA and CountQA).

| Benchmark | Difficulty | SFT (Rank 32) | **GRPO (Rank 64)** | **Improvement** |
| :--- | :--- | :--- | :--- | :--- |
| **TallyQA** | Simple | 76.25% | **80.01%** | 🟢 **+3.76%** |
| **TallyQA** | Complex | 53.71% | **63.75%** | 🟢 **+10.04%** |
| **CountQA** | Hardest | 13.39% | **21.98%** | 🟢 **+8.59%** |

### 3. Impact of LoRA Rank (Robustness)
We analyzed how the LoRA Rank ($R$) affects performance. GRPO remains highly effective even at higher ranks, whereas SFT performance can degrade (likely due to overfitting).

| Model | Strategy | Rank | **Accuracy** | **Reasoning Similarity** |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL** | SFT | $R=32$ | **81.85%** | 99.42% |
| **Qwen2-VL** | SFT | $R=64$ | 70.02% 🔻 | 90.21% |
| | | | | |
| **Qwen2-VL** | GRPO | $R=32$ | **86.97%** | 24.47% |
| **Qwen2-VL** | GRPO | $R=64$ | **85.68%** | 54.58% |

---

## 🧠 Core Philosophy

1.  **Pure Reinforcement Learning:** We do not use "Gold" reasoning traces to train the model. The model only receives a binary reward for the final answer (Accuracy) and a structural reward for the thought format.
2.  **Visual Grounding:** We enforce a "Reasoning Step Reward" that punishes lazy thinking and incentivizes the model to break down visual counting into distinct steps (e.g., "1. Scanning...", "2. Verifying...").
3.  **Format Constraints:** The model is forced to adopt the `<think>...</think><answer>...</answer>` XML structure via a strict regex-based reward.

## 🛠️ Method & Architecture

### The Reward Triad
To guide the model from random guessing to structured reasoning, we use a composite reward function:

* **Format Reward:** +1.0 if the output strictly follows `<think>...</think><answer>...</answer>`.
* **Accuracy Reward:** +1.0 if the numeric answer matches the ground truth.
* **Reasoning Steps Reward:** +0.1 per valid reasoning step (e.g., "1. ", "2. "), capped at 1.0. This prevents "one-line" lazy reasoning.
* **Keyword/Cosine Reward (Optional):** Used in advanced runs to encourage descriptiveness (mentioning colors/shapes) without forcing exact word matching.

## 💻 Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/vision-r1-zero.git](https://github.com/your-username/vision-r1-zero.git)
cd vision-r1-zero

# Create environment
conda create -n vision_rl python=3.10
conda activate vision_rl

# Install dependencies
pip install torch transformers datasets trl peft sentence-transformers bitsandbytes