"""
测试不同 SEED_BASE 下的结果稳健性
证明我们没有 cherry-picking
"""
import numpy as np
from src.extended_dgp import generate_sparse_var_extended
from src.lasso_simulation import compute_lasso_te_matrix

def test_one_config(N=50, T=250, n_trials=20, seed_base=42):
    """测试一个配置下的平均 precision"""
    precisions = []
    
    for trial in range(n_trials):
        seed = seed_base + trial * 1000
        
        # 生成数据
        R, A, A_true = generate_sparse_var_extended(
            N=N, T=T, density=0.05, seed=seed, dgp='garch_factor'
        )
        
        # LASSO 估计
        te_matrix, A_pred = compute_lasso_te_matrix(R)
        A_pred = (A_pred != 0).astype(int)
        
        # 计算 precision
        mask = ~np.eye(N, dtype=bool).flatten()
        yt = A_true.flatten()[mask]
        yp = A_pred.flatten()[mask]
        
        if yp.sum() > 0:
            tp = (yt * yp).sum()
            fp = ((1 - yt) * yp).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            precision = 0
        
        precisions.append(precision)
    
    return np.mean(precisions), np.std(precisions)

# 测试不同的 SEED_BASE
print("Testing robustness across different SEED_BASE values...\n")

seed_bases = [42, 100, 200, 500, 1000]
results = []

for seed_base in seed_bases:
    mean_prec, std_prec = test_one_config(seed_base=seed_base)
    results.append((seed_base, mean_prec, std_prec))
    print(f"SEED_BASE={seed_base:4d}: precision={mean_prec:.4f} ± {std_prec:.4f}")

# 计算变异系数
mean_of_means = np.mean([r[1] for r in results])
std_of_means = np.std([r[1] for r in results])
cv = std_of_means / mean_of_means

print(f"\n--- Stability Analysis ---")
print(f"Mean across SEED_BASEs: {mean_of_means:.4f}")
print(f"Std across SEED_BASEs:  {std_of_means:.4f}")
print(f"Coefficient of Variation: {cv:.2%}")

if cv < 0.05:
    print("\n✅ Results are STABLE (CV < 5%) → No cherry-picking!")
elif cv < 0.10:
    print("\n⚠️ Results are MODERATELY stable (CV < 10%)")
else:
    print("\n❌ Results are UNSTABLE (CV > 10%) → Potential issue")
