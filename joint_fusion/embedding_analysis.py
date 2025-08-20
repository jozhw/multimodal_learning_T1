import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import os
import gc
import joblib
import wandb
from pathlib import Path


def analyze_combined_embeddings(
    embeddings,
    risk_scores,
    num_bins=2,
    epoch=0,
    fold_idx=0,
    checkpoint_dir="checkpoints",
):
    """
    :embeddings: torch.Tensor [n_samples, embedding_dim]
    :risk_scores: torch.Tensor [n_samples]
    :num_bins: Number of risk bins for grouping
    :epoch: Current epoch
    :fold_idx: Current fold
    :checkpoint_dir: Directory to save UMAP plots

    returns: dict: Metrics for logging
    """
    embeddings = embeddings.detach().cpu().numpy()
    risk_scores = risk_scores.detach().cpu().numpy()

    # Bin risk scores into quantiles
    bins = np.quantile(risk_scores, np.linspace(0, 1, num_bins + 1))
    pseudo_labels = np.digitize(risk_scores, bins) - 1  # 0, 1, ..., num_bins-1

    # Compute cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    intra_sim = []
    inter_sim = []

    # Label embeddings
    for i in range(len(pseudo_labels)):
        for j in range(i + 1, len(pseudo_labels)):
            sim = sim_matrix[i, j]
            if pseudo_labels[i] == pseudo_labels[j]:
                intra_sim.append(sim)
            else:
                inter_sim.append(sim)

    # Avg cosine similarity between embedding of samples within the same risk bin
    # Want -> 1.0
    avg_intra = np.mean(intra_sim) if intra_sim else 0
    # Avg cosine similarity between embeddings of samples from different risk bins
    # Want -> 0.0
    avg_inter = np.mean(inter_sim) if inter_sim else 0

    # Correlation with risk differences
    # Want -> -1.0 not -> 1.0
    # Want bigger risk difference with lower cosine similarity
    risk_diffs = np.abs(risk_scores[:, None] - risk_scores[None, :])
    sim_flat = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    diff_flat = risk_diffs[np.triu_indices(len(risk_diffs), k=1)]
    corr = np.corrcoef(sim_flat, diff_flat)[0, 1] if len(sim_flat) > 0 else 0

    # UMAP visualization
    reducer = umap.UMAP()
    embeds_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeds_2d[:, 0], embeds_2d[:, 1], c=risk_scores, cmap="viridis"
    )
    plt.colorbar(label="Risk Score")
    plt.title(f"UMAP of Combined Embeddings (Fold {fold_idx+1}, Epoch {epoch+1})")

    # Save plot
    plot_path = os.path.join(checkpoint_dir, f"umap_fold_{fold_idx}_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()

    # Log to wandb
    wandb.log({f"UMAP Plot {fold_idx}": wandb.Image(plot_path)}, step=epoch)

    metrics = {
        f"CosineSim/IntraBin: {fold_idx}": avg_intra,
        f"CosineSim/InterBin: {fold_idx}": avg_inter,
        f"CosineSim/RiskDiffCorr: {fold_idx}": corr,
    }
    return metrics
