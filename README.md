# MaRCA Codebase

This repository contains the codebase for MaRCA, a framework designed for dynamic computation allocation in large-scale recommender systems. It implements multi-agent reinforcement learning (MARL) techniques to optimize resource usage and maximize business objectives.

## Project Goal

The primary goal of this codebase is to provide a practical implementation of the MaRCA framework, enabling researchers and practitioners to explore, experiment with, and potentially adapt these advanced resource allocation strategies.

## Core Modules & Functionality

The MaRCA framework is built upon three main functional modules, each implemented in its respective directory:

1.  **`1.Adaptive_weighting_recurrent_q-mixer/` - Multi-Agent Reinforcement Learning Engine**
    *   **Functionality:** Implements the core MARL agents responsible for learning optimal resource allocation policies. This includes the AWRQ-Mixer model and various baseline MARL algorithms (DQN, QMIX, VDN, etc.).
    *   **Key Files:**
        *   `main.py`: Entry point for training the MARL models.
        *   `model.py`: Contains the neural network architectures for AWRQ-Mixer and other agents.
        *   `config.py`: Centralized configuration for selecting models, setting hyperparameters, and defining training parameters.
        *   `data.csv`: Sample dataset for training and testing the RL agents.
        *   `Evaluation/`: Contains scripts for model performance analysis.
    *   **Usage Focus:** Training different MARL agents, evaluating their performance in predicting action values (Q-values), and understanding their decision-making processes.

2.  **`2.Autobucket_testbench/` - Computation Cost Estimation**
    *   **Functionality:** Provides tools for estimating the computation cost of different actions within the recommender system. This module employs a machine learning model (MMoE-DCN).
    *   **Key Files:**
        *   `main.py`: Main script to run the cost estimation/prediction process.
        *   `model.py`: Implementation of the cost prediction model.
        *   `config.py`: Configuration for the cost estimation model and data processing.
        *   `data_new.csv`: Sample dataset for training the cost estimation model.
    *   **Usage Focus:** Training a model to accurately predict computation cost, which is a critical input for the resource allocation decision-making process.

3.  **`3.MPC_based_revenue_cost_balancer/` - MPC-based Revenue-Cost Balancer**
    *   **Functionality:** Implements a Model Predictive Control (MPC) system to proactively manage the trade-off between revenue and computation cost. It uses a learned model of the system's dynamics to forecast future states and optimize the control parameter (λ).
    *   **Key Files:**
        *   `main_mpc.py`: Main script to execute the MPC controller.
        *   `mpc_solver.py`: Contains the optimization logic for the MPC.
        *   `system_model.py`: The neural network model that predicts future system states based on current state and actions.
        *   `train.py`: Script for training this predictive `system_model.py`.
        *   `Dataset.py`: Handles data loading and preprocessing for the system model.
        *   `sample_data.csv`: Sample dataset for training the system model and for MPC simulations.
    *   **Usage Focus:** Training a system model and then using the MPC solver to determine optimal control actions (λ values).

4.  **`Revenue_simulation/`**
    *   **Functionality:** A resource allocation simulation system for optimizing computing power distribution between states and actions.
    *   **Key Files:**
        *   `main.py`: Entry point for running these simulations.

## General Project Structure and Workflow (from a Code Perspective)

1.  **Data Preparation:** Each module typically requires its own specific data format. Sample data files are provided in each main module directory.
2.  **Configuration:** Each module has a `config.py` (or uses command-line arguments defined in `utils.py` for MPC) to set model types, hyperparameters, paths, etc. This is the primary way to customize experiments.
3.  **Training Models:**
    *   Train RL agents using `1.Adaptive_weighting_recurrent_q-mixer/main.py`.
    *   Train the cost estimation model using `2.Autobucket_testbench/main.py`.
    *   Train the system model for MPC using `3.MPC_based_revenue_cost_balancer/train.py`.
4. **Running and Simulating:**
    *   Run the MPC controller with a trained system model using `3.MPC_based_revenue_cost_balancer/main_mpc.py`.
    *   The outputs from the AWRQ-Mixer (predicted Q) and AutoBucket TestBench (predicted C) would conceptually feed into the MPC's decision-making process in a full deployment, although the standalone modules allow for independent testing and development.
5. ** Revenue Simulation**
    *   Run the simulation model with a trained model result `4.Revenue_simulation`

## Getting Started

1.  **Clone the repository.**
2.  **Install dependencies:** Check `requirement.txt` (located in the root) and install necessary Python packages.
    ```bash
    pip install -r requirement.txt
    ```
3.  **Explore Individual Modules:**
    *   Navigate into each main directory (`1.Adaptive_weighting_recurrent_q-mixer/`, `2.Autobucket_testbench/`, `3.MPC_based_revenue_cost_balancer/`, `4.Revenue_simulation/`).
    *   Review their specific `README.md` files (if more detailed ones exist beyond what was provided) and `config.py` files.
    *   Start by running the `main.py` or `train.py` scripts with the provided sample data to understand the workflow of each component.

This codebase provides a foundation for understanding and experimenting with advanced techniques for dynamic computation allocation in complex systems. For theoretical underpinnings and experimental results of the MaRCA framework, please refer to the associated research paper.


