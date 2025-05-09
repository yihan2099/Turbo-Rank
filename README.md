# Turbo-Rank

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A high-performance recommendation and ranking system designed for efficient model training, deployment, and serving.

## Project Overview

Turbo-Rank provides a comprehensive framework for building, training, and deploying ranking models with a focus on performance and scalability. It supports various model architectures, including two-tower and NRMS (Neural News Recommendation with Multi-Head Self-Attention), and integrates with MLflow for experiment tracking.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── mind/          <- MIND (Microsoft News Dataset) for recommendation tasks
│   └── processed/     <- The final, canonical data sets for modeling
│
├── deploy/            <- Deployment resources and configuration
│   ├── build/         <- Build artifacts for model deployment
│   ├── docker/        <- Docker configuration for deployment
│   └── scripts/       <- Deployment scripts
│
├── docs               <- Project documentation
│
├── models             <- Trained and serialized models
│   ├── baseline/      <- Baseline model implementations
│   └── onnx/          <- ONNX model exports for deployment
│
├── notebooks          <- Jupyter notebooks for experimentation
│
├── pyproject.toml     <- Project configuration file
│
├── references         <- Data dictionaries, manuals, and explanatory materials
│
├── reports            <- Generated analysis reports
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- The requirements file for reproducing the environment
│
├── server/            <- Serving infrastructure
│   ├── docker/        <- Server Docker configuration
│   └── models/        <- Models for serving
│
├── tests/             <- Test suite for the project
│
└── turbo_rank/        <- Source code for use in this project
    ├── __init__.py    <- Makes turbo_rank a Python module
    ├── cli/           <- Command line interface utilities
    ├── config/        <- Configuration management
    ├── data/          <- Data loading and processing utilities
    ├── datasets/      <- Dataset implementations
    ├── engine/        <- Training and evaluation engines
    ├── evaluation/    <- Evaluation metrics and utilities
    ├── models/        <- Model implementations
    │   ├── nrms.py    <- NRMS model implementation
    │   └── two_tower.py <- Two-tower model implementation
    └── plots.py       <- Visualization utilities
```

## Getting Started

### Prerequisites

- Python 3.10+
- Make
- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Turbo-Rank.git
cd Turbo-Rank

# Create and activate a virtual environment (optional but recommended)
make create_environment

# Install dependencies
make requirements
```

### Usage

Training a model:
```bash
python -m turbo_rank.cli.train --config configs/nrms.yaml
```

## MLflow Integration

The project uses MLflow for experiment tracking. You can view the experiment results by running:

```bash
mlflow ui
```

Then open your browser and navigate to http://localhost:5000.

## License

This project is licensed under the [LICENSE] - see the LICENSE file for details.

--------

