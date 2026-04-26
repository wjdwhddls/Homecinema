"""
paired t-test로 모델 간 통계적 유의성 검정
"""

import numpy as np
from scipy import stats

def compare_models(name1, name2, results_dir="/home/piai/AcousticRooms/xRIR_code-main/results"):
    data1 = np.load(f"{results_dir}/{name1}_per_sample.npz")
    data2 = np.load(f"{results_dir}/{name2}_per_sample.npz")
    
    print(f"\n=== {name1} vs {name2} ===")
    
    for metric in ["edt", "c50", "t60"]:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1[metric], data2[metric])
        
        mean_diff = np.mean(data1[metric] - data2[metric])
        
        # 효과 크기 (Cohen's d)
        diff = data1[metric] - data2[metric]
        cohen_d = np.mean(diff) / np.std(diff)
        
        sig = "✓ 유의" if p_value < 0.05 else "✗ 무의"
        better = name2 if mean_diff > 0 else name1
        
        print(f"{metric.upper()}: "
              f"t={t_stat:.3f}, p={p_value:.4f} [{sig}] "
              f"| {better}이 평균 {abs(mean_diff):.4f} 낮음 "
              f"| Cohen's d={cohen_d:.3f}")


# 모든 조합 비교
compare_models("baseline", "fusion")
compare_models("baseline", "ConvNeXT")