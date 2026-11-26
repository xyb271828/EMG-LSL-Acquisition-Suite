"""Direct supervision of onsite fatigue scores and >=7 subjective-fatigue alerts."""

import copy
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
)
from scipy.special import expit
from scipy.stats import spearmanr
from .data import load_recording
from .features import continuous_features
from .splits import temporal_partitions

SEEDS = [42, 93, 173]


class DirectFatigueNet(nn.Module):
    def __init__(self, extra_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, 15, stride=5, padding=7),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + extra_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 3),
        )

    def forward(self, x, extra):
        z = self.head(torch.cat([self.encoder(x), extra], dim=1))
        return torch.sigmoid(z[:, 0]), torch.clamp(z[:, 1], -7, 0), z[:, 2]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loss_fn(mu, lv, logit, score):
    scaled = score / 10
    nll = 0.5 * (lv + (scaled - mu) ** 2 * torch.exp(-lv))
    return nll.mean() + 0.5 * nn.functional.binary_cross_entropy_with_logits(
        logit, (score >= 7).float()
    )


def train_cnn(X, extra, y, train, val, seed, epochs=None):
    seed_all(seed)
    torch.set_num_threads(2)
    mean = X[train].mean(axis=(0, 2), keepdims=True)
    std = X[train].std(axis=(0, 2), keepdims=True) + 1e-8
    scaler = StandardScaler().fit(extra[train])
    xx = torch.tensor((X - mean) / std, dtype=torch.float32)
    ee = torch.tensor(scaler.transform(extra), dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    model = DirectFatigueNet(extra.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best = float("inf")
    state = None
    best_epoch = 1
    history = []
    stale = 0
    for ep in range(epochs or 120):
        model.train()
        order = np.random.permutation(train)
        total = 0
        for a in range(0, len(order), 16):
            ix = order[a : a + 16]
            mu, lv, logit = model(xx[ix], ee[ix])
            loss = loss_fn(mu, lv, logit, yy[ix])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(ix)
        if val is not None:
            model.eval()
            with torch.no_grad():
                mu, lv, logit = model(xx[val], ee[val])
                prob = torch.sigmoid(logit).numpy()
                pred = 10 * mu.numpy()
            mae = float(mean_absolute_error(y[val], pred))
            nll = float(
                log_loss(y[val] >= 7, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])
            )
            objective = mae / 10 + 0.2 * nll
            history.append(
                dict(
                    epoch=ep + 1,
                    train_loss=total / len(train),
                    validation_mae=mae,
                    validation_alert_nll=nll,
                )
            )
            if objective < best - 1e-5:
                best = objective
                state = copy.deepcopy(model.state_dict())
                best_epoch = ep + 1
                stale = 0
            else:
                stale += 1
            if ep >= 20 and stale >= 20:
                break
    if val is not None:
        model.load_state_dict(state)
    return model, mean, std, scaler, best_epoch, history


def cnn_predict(bundle, X, extra, indices, mc=20):
    model, mean, std, scaler, *_ = bundle
    xx = torch.tensor((X[indices] - mean) / std, dtype=torch.float32)
    ee = torch.tensor(scaler.transform(extra[indices]), dtype=torch.float32)
    model.eval()
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()
    means = []
    variances = []
    probabilities = []
    with torch.no_grad():
        for _ in range(mc):
            mu, lv, logit = model(xx, ee)
            means.append(10 * mu.numpy())
            variances.append(100 * torch.exp(lv).numpy())
            probabilities.append(torch.sigmoid(logit).numpy())
    return (
        np.mean(means, axis=0),
        np.mean(probabilities, axis=0),
        np.var(means, axis=0) + np.mean(variances, axis=0),
    )
