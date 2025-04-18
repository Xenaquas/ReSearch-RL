import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Simulate Exact Match (EM) scores for 500 samples each
n = 500
rag_ppo = np.random.normal(0.624, 0.035, n) * 100  # Convert to percentage
research_grpo = np.random.normal(0.721, 0.028, n) * 100

# Perform two-sample t-test (unequal variances)
t_stat, p_val = ttest_ind(research_grpo, rag_ppo, equal_var=False)
print(f"T-statistic: {t_stat:.2f}, p-value: {p_val:.2e}")

# Bar plot: Mean EM with Standard Deviation
means = [rag_ppo.mean(), research_grpo.mean()]
stds = [rag_ppo.std(ddof=1), research_grpo.std(ddof=1)]

plt.figure()
plt.bar(['RAG + PPO', 'ReSearch (GRPO)'], means, yerr=stds)
plt.ylabel('Exact Match (%)')
plt.title('Mean Exact Match with Standard Deviation')
plt.tight_layout()
plt.show()

# Histogram: Distribution of EM scores
plt.figure()
plt.hist(rag_ppo, bins=20, alpha=0.7, label='RAG + PPO')
plt.hist(research_grpo, bins=20, alpha=0.7, label='ReSearch (GRPO)')
plt.xlabel('Exact Match (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Exact Match Scores')
plt.legend()
plt.tight_layout()
plt.show()


# Print statistical results
print(f"Statistical Results:")
print(f"- RAG+PPO Mean: {rag_scores.mean():.1f}% ± {rag_scores.std():.1f}%")
print(f"- ReSearch Mean: {research_scores.mean():.1f}% ± {research_scores.std():.1f}%")
print(f"\nIndependent t-test (Welch's):")
print(f"t-statistic = {t_stat:.1f}")
print(f"p-value = {p_value:.2e}")

# Interpret results
alpha = 0.05
print("\nInterpretation:")
if p_value < alpha:
    print(f"We reject the null hypothesis (p < {alpha})")
    print("ReSearch (GRPO) significantly outperforms RAG+PPO")
else:
    print(f"We fail to reject the null hypothesis (p ≥ {alpha})")
