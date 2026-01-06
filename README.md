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

## 📊 Experimental Results

Our experiments show that **GRPO-only training consistently outperforms SFT** on quantitative reasoning benchmarks, even without seeing human reasoning traces during training.

### 🏆 Performance Comparison (GRPO vs. SFT)

| Model | Benchmark | SFT (Rank 32) | **GRPO (Rank 64)** | **Improvement** |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL-2B** | **TDIUC** (Quantity) | 81.85% | **86.97%** | 🟢 **+5.1%** |
| | **TallyQA** (Simple) | 76.25% | **80.01%** | 🟢 **+3.8%** |
| | **TallyQA** (Complex) | 53.71% | **63.75%** | 🟢 **+10.0%** |
| | **CountQA** (Hard) | 13.39% | **21.98%** | 🟢 **+8.6%** |
| **SmolVLM-256M**| **TDIUC** (Quantity) | 63.70% | **67.70%** | 🟢 **+4.0%** |
| | **CountQA** (Hard) | 4.00% | **5.49%** | 🟢 **+1.5%** |

> **Note on "Style Drift":** While accuracy improved, the Cosine Similarity to human ground truth dropped significantly (e.g., from 99% in SFT to ~24% in GRPO). This indicates the model developed its **own** reasoning style rather than mimicking the human annotations.


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