# Self-Healing Machine Learning: A Framework for Robust and Adaptive Models

This repository provides the implementation for our work on **Self-Healing Machine Learning**, which explores methods for improving the robustness and reliability of machine learning models in dynamic and unpredictable environments. Our approach employs mechanisms to detect, diagnose, and mitigate issues such as data drift, corruption, and outliers, ensuring that models remain accurate and effective over time.

## Overview

The repository contains scripts and resources necessary to replicate experiments, generate results, and explore the self-healing functionality. The experiments cover various datasets, corruption types, noise levels, and outlier conditions, demonstrating how models can adapt to maintain performance in changing environments.

## Prerequisites

To use this repository, you need an API key for querying an LLM if you plan to use the language model components. You can set this up by creating a Python file named `openai_config.py` within the `src` directory with the following structure:

```python
def get_openai_config():
    openai_config = {
        "api_type": "azure",
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key_main,
        "deployment_id": deployment_name,
        "deployment_id_ada": deployment_name_ada,
        "temperature": 0.0,
        "seed": 0
    }
    return openai_config
```
Replace the placeholders with your actual configuration values.

## Getting Started

### 1. Install Dependencies

Ensure that you have Python 3.10 or later installed. Create a virtual environment and install the required packages by running:

\```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
\```

### 2. Running the Project

- **Run Experiments**: Use the scripts in the `experiments` directory to replicate different studies and generate results. For example:
  ```
  python experiments/study_1.py
  ```

- **Interactive Notebooks**: Open and explore Jupyter notebooks in the `notebooks` directory to visualize results and understand different components:
  ```
  jupyter notebook notebooks/Example_usage.ipynb
  ```

### 3. Citing the paper
If you find this repository useful, please consider citing our work:

```
@article{rauba2024self,
  title={Self-healing machine learning: A framework for autonomous adaptation in real-world environments},
  author={Rauba, Paulius and Seedat, Nabeel and Kacprzyk, Krzysztof and van der Schaar, Mihaela},
  journal={Neural Information Processing Systems (NeurIPS) 2024},
  year={2024}
}
```