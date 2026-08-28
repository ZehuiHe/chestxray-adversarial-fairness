
"""
Shared utilities for the final multi-label ResNet-50 experiments.

This module is used by notebooks 10–13. It deliberately loads the fixed
patient-level multi-label split produced by notebooks 01 and 02 without
rewriting or resampling any split.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


EXPECTED_CORE_LABELS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    """Set the reproducible seed protocol used by every final model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_output_dirs() -> None:
    for folder in ["splits", "checkpoints", "results"]:
        os.makedirs(folder, exist_ok=True)


def load_fixed_multilabel_data(
    config_path: str = "results/multilabel_selected_labels.json",
    train_path: str = "splits/multilabel_train.csv",
    val_path: str = "splits/multilabel_val.csv",
    test_path: str = "splits/multilabel_test.csv",
):
    """Load the fixed splits and audited seven-label list; never recreate a split."""
    with open(config_path, "r", encoding="utf-8") as f:
        audit_config = json.load(f)

    selected_labels = audit_config.get(
        "core_selected_labels", audit_config.get("selected_labels")
    )
    if selected_labels != EXPECTED_CORE_LABELS:
        raise ValueError(
            "The saved core-label list does not match the final seven-label protocol. "
            f"Expected {EXPECTED_CORE_LABELS}, received {selected_labels}."
        )

    label_columns = [f"label_{label}" for label in selected_labels]
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required_cols = ["Image Index", "image_path", "Patient ID", "sex", *label_columns]
    for split_name, split_df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        missing = [c for c in required_cols if c not in split_df.columns]
        if missing:
            raise KeyError(f"{split_name} split is missing required columns: {missing}")
        if split_df["image_path"].isna().any():
            raise ValueError(f"{split_name} split contains missing image paths.")
        if not set(split_df["sex"].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{split_name} split has invalid sex encoding.")

    return train_df, val_df, test_df, selected_labels, label_columns, audit_config


class MultiLabelChestXrayDataset(Dataset):
    """NIH image dataset with metadata retained for patient-cluster bootstrap CIs."""

    def __init__(self, dataframe: pd.DataFrame, label_columns: Sequence[str], transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.label_columns = list(label_columns)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        targets = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        sex = torch.tensor(float(row["sex"]), dtype=torch.float32)
        patient_id = str(row["Patient ID"])
        image_index = str(row["Image Index"])
        return image, targets, sex, patient_id, image_index


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_columns: Sequence[str],
    seed: int,
    batch_size: int = 16,
):
    """Build the shared 224px ImageNet-normalised data protocol."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = MultiLabelChestXrayDataset(
        train_df, label_columns=label_columns, transform=train_transform
    )
    val_dataset = MultiLabelChestXrayDataset(
        val_df, label_columns=label_columns, transform=eval_transform
    )
    test_dataset = MultiLabelChestXrayDataset(
        test_df, label_columns=label_columns, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


def make_disease_criterion(train_df: pd.DataFrame, label_columns: Sequence[str], device):
    positive_counts = train_df[list(label_columns)].sum(axis=0).to_numpy(dtype=np.float32)
    if np.any(positive_counts <= 0):
        raise ValueError("Each final core label must have at least one training positive.")
    negative_counts = len(train_df) - positive_counts
    pos_weight_values = negative_counts / positive_counts
    pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weight_table = pd.DataFrame(
        {
            "label": [c.replace("label_", "", 1) for c in label_columns],
            "train_positive": positive_counts.astype(int),
            "train_negative": negative_counts.astype(int),
            "pos_weight": pos_weight_values,
        }
    )
    return criterion, weight_table, pos_weight_values


def make_r50_baseline(n_labels: int, device):
    """Create ImageNet-pretrained ResNet-50 and dynamically obtain feature_dim."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    feature_dim = model.fc.in_features
    if feature_dim != 2048:
        raise RuntimeError(f"Unexpected ResNet-50 feature dimension: {feature_dim}")
    model.fc = nn.Linear(feature_dim, n_labels)
    model = model.to(device)
    return model, feature_dim, weights


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x, lambda_grl=1.0):
    return GradientReversalFunction.apply(x, lambda_grl)


class MultiLabelAdversarialResNet50(nn.Module):
    """Shared R50 encoder with disease and protected-attribute heads."""

    def __init__(self, n_labels: int):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_dim = base.fc.in_features
        if self.feature_dim != 2048:
            raise RuntimeError(
                f"Unexpected ResNet-50 feature dimension: {self.feature_dim}"
            )

        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.disease_head = nn.Linear(self.feature_dim, n_labels)
        self.sex_head = nn.Linear(self.feature_dim, 1)

    def forward(self, x, lambda_grl=1.0):
        features = torch.flatten(self.encoder(x), 1)
        disease_logits = self.disease_head(features)
        sex_logits = self.sex_head(grad_reverse(features, lambda_grl))
        return disease_logits, sex_logits, features


def safe_auroc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def per_label_ranking_metrics(targets, probs, labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(labels):
        y_true = targets[:, i].astype(int)
        y_prob = probs[:, i]
        rows.append(
            {
                "label": label,
                "support": int(len(y_true)),
                "positive_support": int(y_true.sum()),
                "negative_support": int((1 - y_true).sum()),
                "prevalence": float(y_true.mean()),
                "AUROC": safe_auroc(y_true, y_prob),
                "AUPRC": safe_auprc(y_true, y_prob),
            }
        )
    return pd.DataFrame(rows)


def _collect_prediction_arrays(
    disease_prob_batches,
    target_batches,
    sex_batches,
    patient_id_batches,
    image_index_batches,
):
    return {
        "probs": np.concatenate(disease_prob_batches, axis=0),
        "targets": np.concatenate(target_batches, axis=0),
        "sexes": np.concatenate(sex_batches, axis=0).astype(int),
        "patient_ids": np.asarray(
            [item for batch in patient_id_batches for item in batch], dtype=str
        ),
        "image_indices": np.asarray(
            [item for batch in image_index_batches for item in batch], dtype=str
        ),
    }


def train_baseline_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets, _, _, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return float(total_loss / len(loader.dataset))


def evaluate_baseline(model, loader, criterion, device, labels: Sequence[str]):
    model.eval()
    total_loss = 0.0
    disease_prob_batches, target_batches, sex_batches = [], [], []
    patient_id_batches, image_index_batches = [], []

    with torch.no_grad():
        for images, targets, sexes, patient_ids, image_indices in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)

            disease_prob_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(targets.cpu().numpy())
            sex_batches.append(sexes.numpy())
            patient_id_batches.append(list(patient_ids))
            image_index_batches.append(list(image_indices))

    result = _collect_prediction_arrays(
        disease_prob_batches,
        target_batches,
        sex_batches,
        patient_id_batches,
        image_index_batches,
    )
    label_metrics = per_label_ranking_metrics(result["targets"], result["probs"], labels)
    result.update(
        {
            "loss": float(total_loss / len(loader.dataset)),
            "macro_auroc": float(np.nanmean(label_metrics["AUROC"])),
            "macro_auprc": float(np.nanmean(label_metrics["AUPRC"])),
            "label_metrics": label_metrics,
        }
    )
    return result


def train_adv_one_epoch(
    model,
    loader,
    optimizer,
    disease_criterion,
    sex_criterion,
    device,
    lambda_adv: float,
):
    model.train()
    total_loss = total_disease_loss = total_sex_loss = 0.0
    for images, targets, sexes, _, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        sex_targets = sexes.to(device).unsqueeze(1)
        optimizer.zero_grad()
        disease_logits, sex_logits, _ = model(images, lambda_grl=lambda_adv)
        disease_loss = disease_criterion(disease_logits, targets)
        sex_loss = sex_criterion(sex_logits, sex_targets)
        # λ is applied only through the GRL; do not multiply sex_loss by λ again.
        loss = disease_loss + sex_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_disease_loss += disease_loss.item() * images.size(0)
        total_sex_loss += sex_loss.item() * images.size(0)
    n = len(loader.dataset)
    return {
        "loss": float(total_loss / n),
        "disease_loss": float(total_disease_loss / n),
        "sex_loss": float(total_sex_loss / n),
    }


def dynamic_grl_lambda(
    progress: float,
    lambda_max: float,
    warmup_fraction: float = 0.03,
    ramp_fraction: float = 0.15,
) -> float:
    """Short warm-up followed by a fast cosine ramp to ``lambda_max``.

    This v2 schedule leaves the adversarial signal at zero only during the
    earliest optimisation steps, then increases it smoothly to ``lambda_max``.
    It reaches the maximum at ``warmup_fraction + ramp_fraction`` of total
    training progress and remains there for the rest of training.

    The function is intentionally kept in this *v2* module so that the
    original DANN-logistic schedule used by notebook 05 remains unchanged.
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    warmup_fraction = float(np.clip(warmup_fraction, 0.0, 1.0))
    ramp_fraction = float(ramp_fraction)

    if ramp_fraction <= 0.0:
        raise ValueError("ramp_fraction must be positive.")
    if warmup_fraction + ramp_fraction > 1.0:
        raise ValueError("warmup_fraction + ramp_fraction must not exceed 1.0.")

    if progress <= warmup_fraction:
        return 0.0

    ramp_end = warmup_fraction + ramp_fraction
    if progress >= ramp_end:
        return float(lambda_max)

    phase = (progress - warmup_fraction) / ramp_fraction
    return float(lambda_max * 0.5 * (1.0 - np.cos(np.pi * phase)))


def train_dynamic_adv_one_epoch(
    model,
    loader,
    optimizer,
    disease_criterion,
    sex_criterion,
    device,
    epoch: int,
    global_step: int,
    total_steps: int,
    lambda_max: float,
    warmup_fraction: float,
    ramp_fraction: float,
):
    model.train()
    total_loss = total_disease_loss = total_sex_loss = 0.0
    schedule_rows = []

    for batch_index, (images, targets, sexes, _, _) in enumerate(loader, start=1):
        progress = global_step / max(total_steps - 1, 1)
        lambda_current = dynamic_grl_lambda(
            progress=progress,
            lambda_max=lambda_max,
            warmup_fraction=warmup_fraction,
            ramp_fraction=ramp_fraction,
        )

        images = images.to(device)
        targets = targets.to(device)
        sex_targets = sexes.to(device).unsqueeze(1)
        optimizer.zero_grad()
        disease_logits, sex_logits, _ = model(images, lambda_grl=lambda_current)
        disease_loss = disease_criterion(disease_logits, targets)
        sex_loss = sex_criterion(sex_logits, sex_targets)
        # λ is applied only in the GRL.
        loss = disease_loss + sex_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_disease_loss += disease_loss.item() * images.size(0)
        total_sex_loss += sex_loss.item() * images.size(0)
        schedule_rows.append(
            {
                "epoch": int(epoch),
                "batch": int(batch_index),
                "global_step": int(global_step),
                "progress": float(progress),
                "lambda_adv": float(lambda_current),
            }
        )
        global_step += 1

    n = len(loader.dataset)
    lambda_values = [r["lambda_adv"] for r in schedule_rows]
    return {
        "loss": float(total_loss / n),
        "disease_loss": float(total_disease_loss / n),
        "sex_loss": float(total_sex_loss / n),
        "lambda_start": float(lambda_values[0]),
        "lambda_end": float(lambda_values[-1]),
        "lambda_mean": float(np.mean(lambda_values)),
        "global_step": int(global_step),
        "schedule_rows": schedule_rows,
    }


def evaluate_adversarial(model, loader, disease_criterion, sex_criterion, device, labels):
    model.eval()
    total_loss = total_disease_loss = total_sex_loss = 0.0
    disease_prob_batches, sex_prob_batches, target_batches, sex_batches = [], [], [], []
    patient_id_batches, image_index_batches = [], []

    with torch.no_grad():
        for images, targets, sexes, patient_ids, image_indices in loader:
            images = images.to(device)
            targets = targets.to(device)
            sex_targets = sexes.to(device).unsqueeze(1)
            # GRL has no forward effect; λ=0 clearly documents evaluation-only use.
            disease_logits, sex_logits, _ = model(images, lambda_grl=0.0)
            disease_loss = disease_criterion(disease_logits, targets)
            sex_loss = sex_criterion(sex_logits, sex_targets)
            loss = disease_loss + sex_loss

            disease_prob_batches.append(torch.sigmoid(disease_logits).cpu().numpy())
            sex_prob_batches.append(torch.sigmoid(sex_logits).cpu().numpy().ravel())
            target_batches.append(targets.cpu().numpy())
            sex_batches.append(sexes.numpy())
            patient_id_batches.append(list(patient_ids))
            image_index_batches.append(list(image_indices))
            total_loss += loss.item() * images.size(0)
            total_disease_loss += disease_loss.item() * images.size(0)
            total_sex_loss += sex_loss.item() * images.size(0)

    result = _collect_prediction_arrays(
        disease_prob_batches,
        target_batches,
        sex_batches,
        patient_id_batches,
        image_index_batches,
    )
    result["sex_probs"] = np.concatenate(sex_prob_batches, axis=0)
    label_metrics = per_label_ranking_metrics(result["targets"], result["probs"], labels)
    sex_pred = (result["sex_probs"] >= 0.5).astype(int)
    n = len(loader.dataset)
    result.update(
        {
            "loss": float(total_loss / n),
            "disease_loss": float(total_disease_loss / n),
            "sex_loss": float(total_sex_loss / n),
            "disease_macro_auroc": float(np.nanmean(label_metrics["AUROC"])),
            "disease_macro_auprc": float(np.nanmean(label_metrics["AUPRC"])),
            "sex_auroc": safe_auroc(result["sexes"], result["sex_probs"]),
            "sex_accuracy": float(accuracy_score(result["sexes"], sex_pred)),
            "label_metrics": label_metrics,
        }
    )
    return result


def select_per_label_thresholds(probs, targets, labels, metric: str = "f1"):
    """
    Select each threshold on the full validation set only.
    F1 is the pre-specified validation criterion inherited from the development protocol.
    """
    if metric != "f1":
        raise ValueError("The final protocol fixes threshold metric to validation F1.")
    grid = np.linspace(0.05, 0.95, 181)
    rows = []
    for i, label in enumerate(labels):
        y_true = targets[:, i].astype(int)
        y_prob = probs[:, i]
        best_threshold, best_score = 0.5, -np.inf
        for threshold in grid:
            score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
            if score > best_score:
                best_score, best_threshold = score, float(threshold)
        rows.append(
            {
                "label": label,
                "selected_threshold": float(best_threshold),
                "threshold_metric": "validation_f1",
                "threshold_metric_value": float(best_score),
                "selection_population": "complete_validation_set",
            }
        )
    return pd.DataFrame(rows)


def _threshold_metric_row(label, y_true, y_prob, threshold, group=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    row = {
        "label": label,
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "negative_support": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "AUROC": safe_auroc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "TPR": float(tpr),
        "Recall": float(tpr),  # retained as a transparent alias of TPR
        "FNR": float(fnr),
        "TNR": float(tnr),
        "Specificity": float(tnr),  # retained as a transparent alias of TNR
        "FPR": float(fpr),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    if group is not None:
        row["group"] = group
    return row


def compute_test_metrics_by_group(probs, targets, sexes, labels, thresholds_df):
    """
    Apply validation-derived thresholds unchanged to the full test set and both groups.
    """
    threshold_map = thresholds_df.set_index("label")["selected_threshold"].to_dict()
    overall_rows, subgroup_rows = [], []
    for i, label in enumerate(labels):
        threshold = float(threshold_map[label])
        overall_rows.append(
            _threshold_metric_row(label, targets[:, i], probs[:, i], threshold)
        )
        for sex_value, group in [(0, "Female"), (1, "Male")]:
            mask = np.asarray(sexes).astype(int) == sex_value
            subgroup_rows.append(
                _threshold_metric_row(
                    label,
                    targets[mask, i],
                    probs[mask, i],
                    threshold,
                    group=group,
                )
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(subgroup_rows)


def build_label_fairness_summary(overall_metrics, subgroup_metrics, model_name):
    """
    Fairness summary uses FNR gap once; it does not separately report recall gap,
    because |TPR_F - TPR_M| equals |FNR_F - FNR_M| for binary labels.
    """
    rows = []
    for label in overall_metrics["label"]:
        overall = overall_metrics.loc[overall_metrics["label"] == label].iloc[0]
        female = subgroup_metrics[
            (subgroup_metrics["label"] == label)
            & (subgroup_metrics["group"] == "Female")
        ].iloc[0]
        male = subgroup_metrics[
            (subgroup_metrics["label"] == label)
            & (subgroup_metrics["group"] == "Male")
        ].iloc[0]

        tpr_gap = abs(female["TPR"] - male["TPR"])
        fpr_gap = abs(female["FPR"] - male["FPR"])
        rows.append(
            {
                "model": model_name,
                "label": label,
                "selected_threshold": overall["threshold"],
                "test_AUROC": overall["AUROC"],
                "test_AUPRC": overall["AUPRC"],
                "female_AUROC": female["AUROC"],
                "male_AUROC": male["AUROC"],
                "female_FNR": female["FNR"],
                "male_FNR": male["FNR"],
                "female_FPR": female["FPR"],
                "male_FPR": male["FPR"],
                "FNR_gap": abs(female["FNR"] - male["FNR"]),
                "FPR_gap": float(fpr_gap),
                "Equalized_Odds_gap": float(max(tpr_gap, fpr_gap)),
                "AUROC_gap": abs(female["AUROC"] - male["AUROC"]),
                "AUPRC_gap": abs(female["AUPRC"] - male["AUPRC"]),
                "Worst_group_Recall": min(female["TPR"], male["TPR"]),
                "Worst_group_AUROC": min(female["AUROC"], male["AUROC"]),
            }
        )
    return pd.DataFrame(rows)


def summarise_overall_fairness(label_fairness_summary: pd.DataFrame) -> Dict[str, float]:
    return {
        "mean_FNR_gap": float(label_fairness_summary["FNR_gap"].mean()),
        "mean_FPR_gap": float(label_fairness_summary["FPR_gap"].mean()),
        "mean_Equalized_Odds_gap": float(
            label_fairness_summary["Equalized_Odds_gap"].mean()
        ),
        "mean_Worst_group_Recall": float(
            label_fairness_summary["Worst_group_Recall"].mean()
        ),
        "mean_AUROC_gap": float(label_fairness_summary["AUROC_gap"].mean()),
    }


def save_prediction_archive(path, result: Dict, labels: Sequence[str], adversarial: bool = False):
    payload = {
        "probs": result["probs"],
        "targets": result["targets"],
        "sexes": result["sexes"],
        "patient_ids": np.asarray(result["patient_ids"], dtype=str),
        "image_indices": np.asarray(result["image_indices"], dtype=str),
        "labels": np.asarray(labels, dtype=str),
    }
    if adversarial:
        payload["sex_probs"] = result["sex_probs"]
    np.savez_compressed(path, **payload)


def checkpoint_payload(model, epoch, val_macro_auroc, run_config):
    return {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "val_macro_auroc": float(val_macro_auroc),
        "run_config": run_config,
    }


def base_run_config(
    seed,
    selected_labels,
    feature_dim,
    batch_size,
    num_epochs,
    pos_weight_values,
    checkpoint_path,
):
    return {
        "project_scope": "multi_label_chest_xray_classification",
        "backbone": "ImageNet-pretrained ResNet-50",
        "backbone_weights": "ResNet50_Weights.IMAGENET1K_V2",
        "feature_dim_from_model_fc_in_features": int(feature_dim),
        "selected_labels": list(selected_labels),
        "n_labels": int(len(selected_labels)),
        "split_protocol": "fixed patient-level train/validation/test split loaded from disk",
        "split_files": {
            "train": "splits/multilabel_train.csv",
            "validation": "splits/multilabel_val.csv",
            "test": "splits/multilabel_test.csv",
        },
        "preprocessing": {
            "image_size": [224, 224],
            "train_augmentation": "RandomHorizontalFlip(p=0.5)",
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "seed": int(seed),
        "seed_protocol": "python random, NumPy, PyTorch, DataLoader generator; num_workers=0",
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
        "disease_loss": "BCEWithLogitsLoss with per-label training-set pos_weight",
        "pos_weight_by_label": {
            label: float(value)
            for label, value in zip(selected_labels, pos_weight_values)
        },
        "checkpoint_selection": "maximum validation macro AUROC",
        "threshold_protocol": (
            "one F1-maximising threshold per label on the complete validation set; "
            "fixed unchanged for overall, Female and Male test metrics"
        ),
        "checkpoint": str(checkpoint_path),
        "test_set_hyperparameter_policy": "No threshold or hyperparameter tuning on test data.",
        "bootstrap_unit": "Patient ID cluster resampling from saved test prediction archive",
    }

# ---------------------------------------------------------------------------
# Final multi-seed aggregation and sex-stratified patient-cluster bootstrap
# ---------------------------------------------------------------------------
# These utilities only aggregate completed seed-level results and quantify the
# uncertainty of the fixed test cohort. They do not change model training,
# patient-level splits, validation-derived thresholds, or test predictions.

BOOTSTRAP_PRIMARY_METRICS = [
    "macro_AUROC",
    "macro_AUPRC",
    "mean_FNR_gap",
    "mean_FPR_gap",
    "mean_Equalized_Odds_gap",
    "mean_Worst_group_Recall",
    "mean_AUROC_gap",
]


def aggregate_multi_seed_table(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Return mean, sample standard deviation, and available-seed count.

    Each row of ``df`` should correspond to a completed seed-level result. The
    function preserves the requested grouping columns and creates three columns
    per metric: ``<metric>_mean``, ``<metric>_std``, and ``<metric>_n_seeds``.
    """
    if df.empty:
        return pd.DataFrame(columns=list(group_columns))

    missing_groups = [column for column in group_columns if column not in df.columns]
    if missing_groups:
        raise KeyError(f"Missing aggregation group columns: {missing_groups}")

    present_metrics = [column for column in metric_columns if column in df.columns]
    if not present_metrics:
        return df[list(group_columns)].drop_duplicates().reset_index(drop=True)

    numeric = df[list(group_columns) + present_metrics].copy()
    for metric in present_metrics:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")

    grouped = numeric.groupby(list(group_columns), dropna=False)[present_metrics]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    n_df = grouped.count().add_suffix("_n_seeds")
    return pd.concat([mean_df, std_df, n_df], axis=1).reset_index()


def _metric_summary_from_prediction_arrays(
    probs,
    targets,
    sexes,
    labels: Sequence[str],
    thresholds_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute all pre-specified diagnostic and sex-fairness summaries once."""
    label_ranking = per_label_ranking_metrics(targets, probs, labels)
    overall_metrics, subgroup_metrics = compute_test_metrics_by_group(
        probs=probs,
        targets=targets,
        sexes=sexes,
        labels=labels,
        thresholds_df=thresholds_df,
    )
    fairness_by_label = build_label_fairness_summary(
        overall_metrics=overall_metrics,
        subgroup_metrics=subgroup_metrics,
        model_name="bootstrap_resample",
    )
    fairness_means = summarise_overall_fairness(fairness_by_label)
    return {
        "macro_AUROC": float(np.nanmean(label_ranking["AUROC"])),
        "macro_AUPRC": float(np.nanmean(label_ranking["AUPRC"])),
        **fairness_means,
    }


def _sex_stratified_patient_cluster_indices(sexes, patient_ids, rng):
    """Sample all patient-image clusters with replacement within each sex group.

    Sampling within Female and Male separately keeps both groups represented in
    every bootstrap replicate, which is required for subgroup fairness metrics.
    A patient must have one consistent protected-attribute encoding throughout
    the fixed test split.
    """
    sexes = np.asarray(sexes).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)

    unique_patient_ids, inverse = np.unique(patient_ids, return_inverse=True)
    patient_to_indices = [
        np.flatnonzero(inverse == patient_position)
        for patient_position in range(len(unique_patient_ids))
    ]
    patient_sexes = np.empty(len(unique_patient_ids), dtype=int)

    for patient_position, indices in enumerate(patient_to_indices):
        patient_sex_values = np.unique(sexes[indices])
        if len(patient_sex_values) != 1:
            patient_id = unique_patient_ids[patient_position]
            raise ValueError(
                f"Patient ID {patient_id!r} has inconsistent sex values in the test set."
            )
        patient_sexes[patient_position] = patient_sex_values[0]

    sampled_index_chunks = []
    for sex_value in (0, 1):
        positions = np.flatnonzero(patient_sexes == sex_value)
        if len(positions) == 0:
            group = "Female" if sex_value == 0 else "Male"
            raise ValueError(f"No {group} patients are available for subgroup bootstrap.")
        sampled_positions = rng.choice(positions, size=len(positions), replace=True)
        sampled_index_chunks.extend(patient_to_indices[position] for position in sampled_positions)

    sampled_indices = np.concatenate(sampled_index_chunks)
    return sampled_indices


def patient_cluster_bootstrap_multiseed(
    run_archives: Sequence[Dict],
    n_bootstrap: int = 1000,
    random_seed: int = 202606,
):
    """Compute 95% sex-stratified patient-cluster bootstrap CIs across seeds.

    For every replicate, Female and Male patient IDs are independently sampled
    with replacement. A sampled patient contributes all their test images. The
    identical sampled patient clusters are applied to each seed's prediction
    archive, then the resulting metrics are averaged across seeds.

    Each seed retains its own threshold table selected on the complete
    validation set. No threshold or hyperparameter is selected on the test set.

    Required keys in every item of ``run_archives``:
    ``seed``, ``probs``, ``targets``, ``sexes``, ``patient_ids``,
    ``image_indices``, ``labels``, and ``thresholds_df``.
    """
    if len(run_archives) < 1:
        raise ValueError("At least one completed seed archive is required.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    required_keys = {
        "seed", "probs", "targets", "sexes", "patient_ids",
        "image_indices", "labels", "thresholds_df",
    }
    for archive_index, archive in enumerate(run_archives):
        missing = required_keys.difference(archive.keys())
        if missing:
            raise KeyError(
                f"Run archive {archive_index} is missing required keys: {sorted(missing)}"
            )

    reference = run_archives[0]
    labels = [str(value) for value in np.asarray(reference["labels"]).tolist()]
    targets = np.asarray(reference["targets"])
    sexes = np.asarray(reference["sexes"]).astype(int)
    patient_ids = np.asarray(reference["patient_ids"]).astype(str)
    image_indices = np.asarray(reference["image_indices"]).astype(str)

    n_images = len(targets)
    if targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("Reference target array must be [n_test_images, n_labels].")
    if not (len(sexes) == len(patient_ids) == len(image_indices) == n_images):
        raise ValueError("Reference archive arrays have inconsistent test-image lengths.")

    # Every seed must refer to exactly the same immutable fixed test split and order.
    for archive_index, archive in enumerate(run_archives):
        archive_probs = np.asarray(archive["probs"])
        if archive_probs.shape != targets.shape:
            raise ValueError(
                f"Run archive {archive_index} has probability shape {archive_probs.shape}; "
                f"expected {targets.shape}."
            )
        if [str(value) for value in np.asarray(archive["labels"]).tolist()] != labels:
            raise ValueError("Seed archives have inconsistent label ordering.")
        if not np.array_equal(np.asarray(archive["targets"]), targets):
            raise ValueError("Seed archives have inconsistent test targets.")
        if not np.array_equal(np.asarray(archive["sexes"]).astype(int), sexes):
            raise ValueError("Seed archives have inconsistent sex values.")
        if not np.array_equal(np.asarray(archive["patient_ids"]).astype(str), patient_ids):
            raise ValueError("Seed archives have inconsistent Patient ID ordering.")
        if not np.array_equal(np.asarray(archive["image_indices"]).astype(str), image_indices):
            raise ValueError("Seed archives have inconsistent Image Index ordering.")
        if not isinstance(archive["thresholds_df"], pd.DataFrame):
            raise TypeError("Each run archive must provide thresholds_df as a pandas DataFrame.")

    # Full fixed-test point estimate for each independently trained seed.
    seed_point_rows = []
    for archive in run_archives:
        metrics = _metric_summary_from_prediction_arrays(
            probs=np.asarray(archive["probs"]),
            targets=targets,
            sexes=sexes,
            labels=labels,
            thresholds_df=archive["thresholds_df"],
        )
        metrics["seed"] = int(archive["seed"])
        seed_point_rows.append(metrics)
    seed_point_estimates_df = pd.DataFrame(seed_point_rows)
    mean_point_estimates = (
        seed_point_estimates_df[BOOTSTRAP_PRIMARY_METRICS].mean(axis=0).to_dict()
    )

    rng = np.random.default_rng(random_seed)
    distribution_rows = []
    for bootstrap_iteration in range(n_bootstrap):
        sampled_indices = _sex_stratified_patient_cluster_indices(
            sexes=sexes,
            patient_ids=patient_ids,
            rng=rng,
        )

        per_seed_rows = []
        for archive in run_archives:
            per_seed_rows.append(
                _metric_summary_from_prediction_arrays(
                    probs=np.asarray(archive["probs"])[sampled_indices],
                    targets=targets[sampled_indices],
                    sexes=sexes[sampled_indices],
                    labels=labels,
                    thresholds_df=archive["thresholds_df"],
                )
            )

        mean_metrics = (
            pd.DataFrame(per_seed_rows)[BOOTSTRAP_PRIMARY_METRICS].mean(axis=0).to_dict()
        )
        distribution_rows.append(
            {
                "bootstrap_iteration": int(bootstrap_iteration),
                "n_images_in_resample": int(len(sampled_indices)),
                "n_seeds_aggregated": int(len(run_archives)),
                "resampling_unit": "Patient ID cluster (stratified by sex)",
                **{
                    metric: float(mean_metrics[metric])
                    for metric in BOOTSTRAP_PRIMARY_METRICS
                },
            }
        )

    distribution_df = pd.DataFrame(distribution_rows)
    ci_rows = []
    for metric in BOOTSTRAP_PRIMARY_METRICS:
        values = distribution_df[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            ci_lower = ci_upper = bootstrap_mean = np.nan
        else:
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            bootstrap_mean = float(np.mean(values))
        ci_rows.append(
            {
                "metric": metric,
                "point_estimate_mean_across_seeds": float(mean_point_estimates[metric]),
                "bootstrap_mean": bootstrap_mean,
                "ci_level": 0.95,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n_bootstrap": int(n_bootstrap),
                "n_seeds_aggregated": int(len(run_archives)),
                "resampling_unit": "Patient ID cluster (stratified by sex)",
                "threshold_policy": (
                    "fixed seed-specific validation-derived per-label thresholds; "
                    "no test-set threshold selection"
                ),
            }
        )

    return distribution_df, pd.DataFrame(ci_rows), seed_point_estimates_df
