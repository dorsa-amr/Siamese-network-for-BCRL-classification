"""Statistical comparison of the AUC of the Unimodal-B and Multimodal models (Supplementary Material, Sec. II).

The AUC values are the per-run (averaging strategy) and per-fold (ensembling strategy) results printed by
evaluate.py for the two models.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Averaging strategy: one AUC per fold and seed (Supplementary Table II)
df = pd.DataFrame({
    "fold": [1] * 5 + [2] * 5 + [3] * 5,
    "seed": [42, 43, 44, 45, 46] * 3,
    "auc_B": [0.7619, 0.7262, 0.7976, 0.7857, 0.8810,
              0.5909, 0.9773, 0.8409, 0.3750, 0.9205,
              0.9524, 0.9048, 0.7738, 0.9405, 0.5238],  # Unimodal-B
    "auc_M": [0.8417, 0.8229, 0.8106, 0.8179, 0.8422,
              0.9349, 0.9171, 0.8754, 0.9120, 0.8778,
              0.9592, 0.9430, 0.9335, 0.9020, 0.9575],  # Multimodal
})
df["diff"] = df["auc_M"] - df["auc_B"]  # positive: Multimodal is better

# Ensembling strategy: one AUC per fold
auc_B_ensemble = np.array([0.833, 0.966, 0.940])
auc_M_ensemble = np.array([0.833, 0.931, 0.965])


def blocked_signflip_pvalue(df, n_perm=50000, seed=0):
    """Two-sided sign-flip permutation test of the mean difference, with signs flipped within each fold
    to account for the dependence between seeds that share the same validation fold."""
    rng = np.random.default_rng(seed)
    diffs, folds = df["diff"].to_numpy(), df["fold"].to_numpy()
    observed = diffs.mean()
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        flipped = diffs.copy()
        for f in np.unique(folds):
            idx = np.where(folds == f)[0]
            flipped[idx] *= rng.choice([-1, 1], size=len(idx))
        perm_means[i] = flipped.mean()
    return observed, np.mean(np.abs(perm_means) >= abs(observed))


if __name__ == "__main__":
    print("Averaging strategy (N runs = %d)" % len(df))
    t_stat, p_t = stats.ttest_rel(df["auc_M"], df["auc_B"])
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
    w_stat, p_w = stats.wilcoxon(df["diff"])
    print(f"  Wilcoxon signed-rank: W = {w_stat:.3f}, p = {p_w:.4f}")
    observed, p_block = blocked_signflip_pvalue(df)
    print(f"  Mean AUC difference (M - B) = {observed:.4f}, fold-blocked sign-flip permutation p = {p_block:.4f}")

    print("Ensembling strategy (N folds = 3)")
    diff = auc_M_ensemble - auc_B_ensemble
    print(f"  Differences: {diff}, mean = {diff.mean():.4f}")
    t_stat_e, p_t_e = stats.ttest_rel(auc_M_ensemble, auc_B_ensemble)
    print(f"  Paired t-test: t = {t_stat_e:.3f}, p = {p_t_e:.4f}")
    p_sign = stats.binomtest(int(np.sum(diff > 0)), n=len(diff), p=0.5, alternative='greater').pvalue
    print(f"  Exact sign test: p = {p_sign:.4f}")

    p_adj = multipletests([p_block, p_t_e], method="holm")[1]
    print(f"Holm-adjusted p-values (averaging, ensembling): {p_adj}")
