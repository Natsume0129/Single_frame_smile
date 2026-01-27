import torch
import matplotlib.pyplot as plt
import numpy as np

data = torch.load("compare_out/raw_metrics.pt")

# ---------- 1. Frame-by-frame cosine distance ----------
dcos = data["dcos"].numpy()

plt.figure(figsize=(6,3))
plt.plot(dcos, linewidth=1)
plt.xlabel("Frame index")
plt.ylabel("Cosine distance")
plt.title("Frame-by-frame feature difference (bg vs green)")
plt.tight_layout()
plt.show()

# ---------- 2. kNN overlap ----------
knn = data["knn_overlap"]
if knn is not None:
    knn = knn.numpy()
    plt.figure(figsize=(4,3))
    plt.hist(knn, bins=20)
    plt.xlabel("kNN overlap")
    plt.ylabel("Count")
    plt.title("Structural consistency within segment")
    plt.tight_layout()
    plt.show()

# ---------- 3. Temporal smoothness ----------
bg = data["smooth_bg"].numpy()
rvm = data["smooth_rvm"].numpy()

plt.figure(figsize=(4,3))
plt.boxplot([bg, rvm], labels=["BG", "Green"])
plt.ylabel("||f(t) - f(t+1)||")
plt.title("Temporal smoothness comparison")
plt.tight_layout()
plt.show()
