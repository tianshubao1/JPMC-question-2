# Particle Flows for State-Space Models  
### Implementation for: **Part 2 – Stochastic Flows & Differentiable Particle Filters**

This repository implements the required assignments in **Part 2** of Question 2, covering Stochastic Particle Flows (Dai, 2022) and Differentiable Particle Filters (Soft vs. Optimal Transport)

---

## Requirements

Create a file named `requirements.txt`:

```txt
matplotlib==3.10.7
numpy==2.1.3
pytest==8.4.2
scipy==1.16.3
tensorflow==2.19.1
```
## Folder Structure

All Part 2 code lives under part 2/Code_w_unit_test/

All unit tests are inside in part 2/Code_w_unit_test/tests/.
```
.
├── requirements.txt
└── part 2/
    └── Code_w_unit_test/
        ├── part2_stochastic_flow.py       # Part 2(1)(a): Dai (22) Stiffness Mitigation
        ├── part2_LEDH_improvement.py      # Part 2(1)(b): Li (17) vs Dai (22) Comparison
        ├── part2_soft_dpf.py              # Part 2(i)(a): Differentiable PF (Soft Resampling)
        ├── part2_sinkhorn_ot.py           # Part 2(i)(b): Differentiable PF (OT/Sinkhorn)
        ├── part2_ii_compare.py            # Part 2(ii): Benchmark (Soft vs OT vs Neural)
        └── tests/                         # All unit tests for Part 2
            ├── test_part2_stochastic_flow.py
            ├── test_part2_LEDH_improvement.py
            ├── test_part2_soft_dpf.py
            ├── test_part2_sinkhorn_ot.py
            └── test_part2_ii_compare.py
```



 ## Run the main scripts and unit tests

 **We strongly recommend using a clean Conda environment to avoid binary incompatibility issues (NumPy / TensorFlow / h5py ABI conflicts).**
 
Create and activate the environment:
```
conda create -n jpmc_q2 python=3.12 -y
conda activate jpmc_q2
```
Install required packages:
```
pip install -r requirements.txt
```

Navigate to the source code directory:
```
cd "part 2/Code_w_unit_test"
```

Run Main Experiments

1. Stochastic Flow (Dai 22)
```
# Replicate Dai(22) results (Stiffness, BVP, SDE)
python part2_stochastic_flow.py

# Compare Li(17) Baseline vs Dai(22) Enhanced Flow
python part2_LEDH_improvement.py
```

2. Differentiable Particle Filters
```
# Run Soft Resampling DPF Demo
python part2_soft_dpf.py

# Run Sinkhorn OT Resampling Demo
python part2_sinkhorn_ot.py

# Run Full Comparison Benchmark (Soft vs OT vs Neural)
python part2_ii_compare.py
```

Run All Unit Tests
This will verify BVP solvers, SDE drifts, gradients, and resampling logic.
```
pytest -q
```
