# ENGR 6991 Final Project — Adversarial Robustness of TAM for Graph Anomaly Detection

**Ismail Mzouri | Concordia University | ENGR 6991**

## Overview

This project extends [TAM (Truncated Affinity Maximization, NeurIPS 2023)](https://arxiv.org/pdf/2306.00006.pdf) by investigating its adversarial robustness. We design a **relation camouflage attack** that exploits TAM's NSGT mechanism, propose a **Jaccard preprocessing defense**, and investigate the directional relationship between graph clustering coefficient and attack vulnerability.

## Key Contributions

- **Relation Camouflage Attack**: for each anomalous node, inject k edges to its most cosine-similar normal neighbors, bypassing TAM's NSGT filter
- **Jaccard Defense**: remove edges with low neighborhood Jaccard similarity before training to filter injected edges
- **Structural Analysis**: graph clustering coefficient shows a directional trend with attack vulnerability (Pearson r=0.76, p=0.24, n=4 datasets)

## Results

| Dataset | Clean AUROC | Attacked (k=5) | Defended (τ=0.1) | Clustering Coeff |
|---------|-------------|----------------|------------------|-----------------|
| Amazon | 0.711 | 0.500 (−0.211) | 0.635 (+0.135) | 0.691 |
| Facebook | 0.777 | 0.423 (−0.355) | 0.618 (+0.196) | 0.527 |
| Reddit | 0.581 | 0.496 (−0.083) | N/A (graph collapse) | 0.000 |
| BlogCatalog | 0.670 | 0.658 (−0.012) | N/A (sparse graph) | 0.123 |

## Repository Structure
```
├── train_camouflage_v2.py   # Main script: attack + Jaccard defense
├── train_camouflage.py      # Attack only (v1)
├── summarize_camouflage.py  # Summarize and print results
├── generate_figures.py      # Generate all 4 publication figures
├── model.py                 # TAM model implementation
├── utils.py                 # Utility functions
├── figures/                 # Generated figures (PNG + PDF)
├── results_camouflage_v2_*.npy  # Main experiment results
├── results_camouflage_*.npy     # V1 attack results
└── Amazon_TAM.ipynb         # Exploratory notebook
```

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run attack + defense experiment:**
```bash
python train_camouflage_v2.py --dataset Amazon --k 5 --tau 0.1
```

**Summarize results:**
```bash
python summarize_camouflage.py
```

**Generate figures:**
```bash
python generate_figures.py
```

## Datasets

Download from [TAM original repo](https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z) and place `.mat` files in `data/`.

## Environment

- Python 3.12, PyTorch, CUDA 11.8, NVIDIA RTX 3080 Ti 12GB

## Base Model

TAM: https://github.com/mala-lab/TAM-master

## Citation
```bibtex
@inproceedings{qiao2023truncated,
  title={Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection},
  author={Qiao, Hezhe and Pang, Guansong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
