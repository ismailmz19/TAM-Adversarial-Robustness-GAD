"""
Summarize relation camouflage attack results across all datasets.
Usage: python summarize_camouflage.py
"""
import os
import numpy as np

DATASETS = ['Amazon', 'Facebook', 'Reddit', 'BlogCatalog', 'ACM']

print("\n" + "="*90)
print("  RELATION CAMOUFLAGE ATTACK ON TAM — SUMMARY")
print("="*90)
print(f"  {'Dataset':<14} {'k':>4}  {'Base AUROC':>12}  {'Def AUROC':>12}  {'Δ vs clean':>12}  {'Def Δ vs base':>15}")
print("  " + "-"*86)

for ds in DATASETS:
    fname = f'results_camouflage_{ds}.npy'
    if not os.path.exists(fname):
        print(f"  {ds:<14}  [not found]")
        continue

    data = np.load(fname, allow_pickle=True).item()
    k_values = data['k_values']
    results  = data['results']

    k0 = k_values[0]
    clean_auroc = results[k0]['auroc_base']

    for k in k_values:
        r = results[k]
        delta_vs_clean = r['auroc_base'] - clean_auroc
        delta_def      = r['auroc_defense'] - r['auroc_base']
        sign_c = '+' if delta_vs_clean >= 0 else ''
        sign_d = '+' if delta_def >= 0 else ''
        label = ds if k == k0 else ''
        print(f"  {label:<14} {k:>4}  "
              f"{r['auroc_base']:>12.4f}  "
              f"{r['auroc_defense']:>12.4f}  "
              f"{sign_c}{delta_vs_clean:>11.4f}  "
              f"{sign_d}{delta_def:>14.4f}")
    print()

print("="*90 + "\n")
