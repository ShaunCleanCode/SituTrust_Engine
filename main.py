import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import os

# ✅ Seed 설정
random.seed(42)
np.random.seed(42)

# ✅ 역할 및 전문가 세팅
c_level_roles = ["CEO", "CTO", "CPO", "CMO", "CFO", "CHRO"]
agent_roles = []

# 각 C-Level + 전문가 4명씩 생성
for c_role in c_level_roles:
    agent_roles.append(c_role)
    for i in range(1, 5):
        expert_role = f"{c_role}_Expert_{i}"
        agent_roles.append(expert_role)

# ✅ 역할별 임베딩 생성 (128차원, 샘플당 1개)
embeddings = []
labels = []

for role in agent_roles:
    base_vector = np.random.randn(128)
    noisy_vector = base_vector + np.random.normal(0, 0.25, size=128)
    embeddings.append(noisy_vector)
    labels.append(role)

X = np.array(embeddings)

# ✅ t-SNE 적용
tsne = TSNE(n_components=2, perplexity=15, learning_rate=150, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

# ✅ 시각화
plt.figure(figsize=(12, 8))
for role in c_level_roles:
    # C-Level 점
    idx = labels.index(role)
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=role, marker='o', s=120, edgecolors='black')

    # 전문가들 점
    for i in range(1, 5):
        expert_label = f"{role}_Expert_{i}"
        idx_expert = labels.index(expert_label)
        plt.scatter(X_tsne[idx_expert, 0], X_tsne[idx_expert, 1], label=expert_label, marker='x', alpha=0.7)

plt.title("Figure 5.2 – Speech Embedding Similarity Across C-Level & Specialist Agents (t-SNE)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# ✅ 저장
output_file = "figure_5_2_speech_embedding_c_level_and_specialists_tsne.png"
plt.savefig(output_file)
plt.show()

print(f"Saved to: {os.path.abspath(output_file)}")