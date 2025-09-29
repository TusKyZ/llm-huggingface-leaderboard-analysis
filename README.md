# Mini_Project: A Large Language Model Analysis from the HuggingFace Leaderboard 2024-2025

# Introduction

Large Language Model, like ChatGPT, Claude, Deepseek, and many others have been popular and an increasing trend since 2022, they will continue to be a part of our everyday's life and it is crucial to know how they work and the keywords associated with them. Their core keywords like the parameters, benchmark scores, models fine-tuning, and techniques like Mixture of Experts (MoE), which is used for efficiency and how its important will be explored on as well.

This project aims to explore the HuggingFace leaderboard dataset, targeting the parameters, the average benchmark scores of MoE models vs non-MoE models, and the CO2 costs of the models and visualizing them to find insights.

# Table of Contents

Key Concepts and Terminology

Data Preparation

Insights/Graphs

Conclusion

# Key Concepts and Terminology

This section will explain the keywords that will be used in this project.

Parameters: Parameters refer to the numerical values within a large language model that are learned during training. They determine how the model processes and generates text by adjusting the weights and biases of its neural network. In general, a higher number of parameters increases a model’s capacity to capture complex patterns in data, although this does not always guarantee improved performance or efficiency.

Benchmark Scores: Benchmark scores represent standardized performance evaluations of large language models across a variety of tasks, such as reasoning, language understanding, and problem-solving. They provide a common ground for comparing different models by aggregating results from widely recognized datasets and tests. Average benchmark scores are often used to summarize overall capability, though specific benchmarks may highlight strengths and weaknesses in particular domains.

Mixture of Experts (MoE): The Mixture of Experts (MoE) is an architectural approach designed to improve efficiency in large models. Instead of activating all parts of the model simultaneously, MoE structures contain multiple specialized “experts,” with only a subset being used for each input. This allows models to scale to very large sizes while maintaining lower computational costs during inference, balancing performance and efficiency.

Fine-tuning: Fine-tuning is the process of adapting a pre-trained language model to specific tasks or domains by continuing its training on targeted datasets. This technique allows models to specialize beyond their general training, improving relevance and accuracy in specific tasks.

# Data Preparation

The dataset is from the HuggingFace Leaderboard Dataset
Retrieved through:
from datasets import load_dataset
dataset = load_dataset("open-llm-leaderboard/contents", split="train")
