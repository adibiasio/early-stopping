# Early Stopping Simulation [![Python Versions](https://img.shields.io/badge/python-3.10-blue)]()

This project is dedicated towards identifying efficient early stopping strategies in iterative learners, primarily by exploring adaptive patience functions instead of constant thresholds commonly used in machine learning libraries. 

Research [findings](https://github.com/user-attachments/files/26095393/andrew-final-2024.pdf) were conducted during an internship with the AutoGluon team at AWS AI.


## Early Stopping Strategies

The following stopping mechanisms were used in various combinations to explore a set of early stopping strategies:

<img width="4235" height="2056" alt="background" src="https://github.com/user-attachments/assets/ce165ab5-b89a-410e-a4d1-04f3e5cc42f3" />

<img width="4235" height="2285" alt="simple-and-linear" src="https://github.com/user-attachments/assets/1b3c62c8-95b6-4a19-85fc-d22c7710f5c3" />

<img width="4125" height="2224" alt="polynomial-and-feature" src="https://github.com/user-attachments/assets/f8fadc1e-7944-4d29-b705-3d96291fa559" />

<img width="4125" height="2030" alt="minimum-and-maximum" src="https://github.com/user-attachments/assets/50f0dece-b6a9-4b3e-b58a-4e9281677f0c" />

<img width="4125" height="2030" alt="min-delta-and-sliding-window" src="https://github.com/user-attachments/assets/2275a764-2ff7-4acc-8022-5fcb61edcdb2" />

<img width="3388" height="1802" alt="strategies" src="https://github.com/user-attachments/assets/4e48e9a7-f8eb-4771-a167-f3c1c655cbcd" />


## Benchmarking and Results

The strategies were evaluated on static learning curve data produced with [TabArena](https://github.com/autogluon/tabarena) across 104 datasets. Using these learning curves, we simulated the training process with the following stopping strategies:

1. AutoGluon's Implementation of Adaptive Patience (Baseline)
2. Simple, Linear, Polynomial, and Feature Patience
3. Minimum Delta, Sliding Window, and Fixed Iteration

Note that each strategy also featured variations with minimum/maximum patience bounds. After benchmarking, each strategy was then ranked first on the difference in model performance and second on the difference in training iterations relative to the global optimal iteration.

![benchmarking](https://github.com/user-attachments/assets/0cc228af-ff02-4e6e-b99d-1cae7214b50e)


## Installation Steps

Step 1: Clone this repository and navigate to its root directory

```
git clone https://github.com/adibiasio/early-stopping.git
cd early-stopping
```

Step 2: Make a virtual environment and activate it
```
python -m venv .venv
source .venv/bin/activate
```

Step 3: Source install
```
pip install -e .
```

Step 4: Run the sample script!
```
python examples/sample_run.py
```
