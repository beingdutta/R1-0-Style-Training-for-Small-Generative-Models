# 👁️ Vision-R1-Zero: Self-Evolving Quantitative Reasoning in VLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![TRL](https://img.shields.io/badge/TRL-GRPO-green)

> **"Reasoning should emerge, not be imitated."**

## 🚀 Overview

This project explores the application of **DeepSeek R1-Zero** style training to **Vision Language Models (VLMs)**. We demonstrate that small VLMs (256M - 2B parameters) can self-evolve **Chain-of-Thought (CoT)** reasoning capabilities for quantitative tasks using **only** Group Relative Policy Optimization (GRPO), completely **skipping the Supervised Fine-Tuning (SFT) phase**.

By designing a robust reward signal, we force the model to "think" before it speaks, analyzing visual elements step-by-step to improve counting and quantitative reasoning accuracy.

## 🧠 Core Philosophy: The R1-Zero Approach

Traditional VLM training follows a pipeline: `Pretraining -> SFT -> RLHF`.
This project tests the hypothesis: **Can we skip SFT?**

* **❌ No Human Demonstrations:** The model is not taught *how* to reason via SFT datasets.
* **✅ Pure Reinforcement Learning:** The model starts with a "Cold Start" and learns to reason solely by maximizing a complex reward function.
* **✅ Visual Grounding:** The reasoning chains are forced to be grounded in visual features (color, shape, position).

## 🛠️ Method & Architecture

### Supported Models
* **Qwen2-VL-2B-Instruct** (The robust baseline)
* **SmolVLM-256M-Instruct** (The lightweight challenger)

### The Reward Triad 🏆
We utilize a multi-signal reward function to guide the model from "random guessing" to "structured reasoning":

1.  **Format Reward:** Strictly enforces the `<think>...</think><answer>...</answer>` XML structure.
2.  **Accuracy Reward:** Binary reward for getting the final count correct.
3.  **Reasoning Steps Reward:** Incentivizes distinct logical steps (e.g., "1. Scanning...", "2. Verifying...") to prevent "lazy" one-line thoughts.
4.  **Keyword Recall & Descriptiveness:** (Optional) Rewards the usage of visual attribute keywords found in ground truth to encourage detailed "looking."

## 📊 Experimental Results

We observed distinct phases of learning during the GRPO process:

1.  **The Cold Start:** Initially, the model struggles to format tags, receiving 0 reward.
2.  **The "Aha!" Moment:** Once the model accidentally generates valid tags, the Format Reward spikes.
3.  **Length Collapse vs. Expansion:** Without step-based rewards, the model optimizes for brevity (short, lazy thoughts). With our **Step Reward**, the model learns to generate detailed, multi-step verification traces.

| Model | Technique | SFT Phase? | Accuracy | Reasoning Quality |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL-2B** | GRPO (LoRA) | **No** | 🟢 High | Detailed, Structured |
| **SmolVLM-256M**| GRPO (LoRA) | **No** | 🟡 Moderate | Requires "Force Start" Prompting |

## 💻 Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/vision-r1-zero.git](https://github.com/your-username/vision-r1-zero.git)
cd vision-r1-zero

# Create environment
conda create -n vision_rl python=3.10
conda activate vision_rl

# Install dependencies (ensure you have trl and peft)
pip install torch transformers datasets trl peft sentence-transformers