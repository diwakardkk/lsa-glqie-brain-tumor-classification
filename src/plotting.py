"""Publication-quality plotting and validation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from .allocation import allocation_profiles
from .dataset_io import BrainTumorDataset, build_metadata
from .quantum_simulator import logical_resource_counts


FIGURE_NAMES = [
    "dataset_overview",
    "method_workflow",
    "lesion_size_and_allocation_analysis",
    "main_performance_comparison",
    "confusion_and_per_class_results",
    "lesion_size_subgroup_and_allocation_ablation",
    "robustness_and_resource_tradeoffs",
    "roc_pr_and_calibration_summary",
]

CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
METHOD_NAMES = {
    "global_only": "Global only",
    "local_only": "Local only",
    "fixed_2g_6l": "Fixed local-heavy",
    "fixed_4g_4l": "Fixed balanced",
    "fixed_6g_2l": "Fixed global-heavy",
    "random_allocation": "Random allocation",
    "proposed_lsa_glqie": "LSA-GLQIE",
    "deep_quantum_fusion": "LSA-DQF",
    "classical_proposed": "Classical LR LSA",
    "classical_fixed_4g_4l": "Classical LR fixed",
    "svm_classical_proposed": "Classical SVM LSA",
    "svm_classical_fixed_4g_4l": "Classical SVM fixed",
    "lesion_size_only": "Size only",
    "shuffled_size_allocation": "Shuffled size",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000", "#7F7F7F", "#8B5A2B", "#4B0082"]


def _save_and_validate(fig: plt.Figure, output_dir: Path, number: int, name: str, cfg: dict) -> dict[str, object]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"figure_{number}_{name}.png"
    pdf = plot_dir / f"figure_{number}_{name}.pdf"
    dpi = int(cfg["plots"]["dpi"])
    min_w = int(cfg["plots"]["png_min_width"])
    min_h = int(cfg["plots"]["png_min_height"])
    for attempt in range(3):
        fig.canvas.draw()
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        with Image.open(png) as im:
            arr = np.asarray(im.convert("L"))
            ok = im.width >= min_w and im.height >= min_h and png.stat().st_size >= 20_000 and float(arr.var()) > 0
            if ok:
                plt.close(fig)
                return {"figure": number, "name": name, "png": str(png), "pdf": str(pdf), "width": im.width, "height": im.height, "file_size": png.stat().st_size, "pixel_variance": float(arr.var()), "valid": True}
        fig.set_size_inches(fig.get_size_inches()[0] * 1.2, fig.get_size_inches()[1] * 1.2)
    plt.close(fig)
    raise RuntimeError(f"Figure validation failed for {name}")


def _write_plot_data(output_dir: Path, number: int, name: str, dataframes: dict[str, pd.DataFrame]) -> None:
    """Save machine-readable source data used to construct a final figure."""
    data_dir = output_dir / "plots" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for suffix, df in dataframes.items():
        safe_suffix = suffix.replace(" ", "_").lower()
        df.to_csv(data_dir / f"figure_{number}_{name}_{safe_suffix}.csv", index=False)


def _display_methods(methods: pd.Series | list[str]) -> list[str]:
    return [METHOD_NAMES.get(str(m), str(m).replace("_", " ")) for m in methods]


def _metric_summary(fold_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        vals = group[metric].dropna().astype(float).to_numpy() if metric in group else np.array([])
        if vals.size:
            mean = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            rows.append({"method": method, "method_label": METHOD_NAMES.get(method, method), "mean": mean, "ci95": ci, "n": vals.size})
    return pd.DataFrame(rows).sort_values("mean", ascending=True)


def _plot_barh(ax: plt.Axes, summary: pd.DataFrame, title: str, xlabel: str) -> None:
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(summary))]
    ax.barh(summary["method_label"], summary["mean"], xerr=summary["ci95"], color=colors, alpha=0.88, capsize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(1.0, float((summary["mean"] + summary["ci95"]).max()) * 1.08) if not summary.empty else 1.0)
    ax.grid(axis="x", alpha=0.25)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(float)
    nz = x[x > 0]
    vals = nz if nz.size else x.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _example_samples(data_dir: str | Path | None) -> list[dict[str, object]]:
    if data_dir is None or not Path(data_dir).exists():
        return []
    md = build_metadata(data_dir)
    dataset = BrainTumorDataset(data_dir, md)
    examples = []
    for label in [1, 2, 3]:
        row = md[md["class_label"] == label].iloc[0]
        sample = dataset.read_sample(int(row["file_number"]))
        examples.append(sample)
    return examples


def _overlay_mask(ax: plt.Axes, sample: dict[str, object], title: str) -> None:
    image = _normalize_image(np.asarray(sample["image"]))
    mask = np.asarray(sample["tumorMask"]) > 0
    ax.imshow(image, cmap="gray")
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[..., 0] = 0.95
    overlay[..., 1] = 0.25
    overlay[..., 3] = mask.astype(float) * 0.35
    ax.imshow(overlay)
    border = np.asarray(sample.get("tumorBorder", [])).ravel()
    if border.size >= 4:
        # Swap x and y to match Python's row-major (transposed) MATLAB loader format
        xs = border[1::2]
        ys = border[0::2]
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color="#F0E442", linewidth=1.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _primary_method(cfg: dict, fold_metrics: pd.DataFrame) -> str:
    configured = cfg.get("plots", {}).get("primary_method")
    if configured and configured in set(fold_metrics["method"].astype(str)):
        return str(configured)
    if "proposed_lsa_glqie" in set(fold_metrics["method"].astype(str)):
        return "proposed_lsa_glqie"
    if not fold_metrics.empty:
        metric = "patient_macro_f1" if "patient_macro_f1" in fold_metrics else "slice_macro_f1"
        means = fold_metrics.groupby("method")[metric].mean()
        return str(means.idxmax())
    return "proposed_lsa_glqie"


def _best_baseline(fold_metrics: pd.DataFrame, primary_method: str) -> str:
    candidates = [m for m in fold_metrics["method"].unique() if m != primary_method]
    if not candidates:
        return primary_method
    metric = "patient_macro_f1" if "patient_macro_f1" in fold_metrics else "slice_macro_f1"
    means = fold_metrics[fold_metrics["method"].isin(candidates)].groupby("method")[metric].mean()
    return str(means.idxmax())


def _plot_confusion(ax: plt.Axes, pred: pd.DataFrame, method: str, title: str) -> pd.DataFrame:
    data = pred[pred["method"] == method]
    cm = confusion_matrix(data["true_label"], data["pred_label"], labels=[1, 2, 3])
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([0, 1, 2], [CLASS_NAMES[i] for i in [1, 2, 3]], rotation=20, ha="right")
    ax.set_yticks([0, 1, 2], [CLASS_NAMES[i] for i in [1, 2, 3]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for y in range(3):
        for x in range(3):
            ax.text(x, y, f"{cm[y, x]}\n{norm[y, x]:.2f}", ha="center", va="center", color="white" if norm[y, x] > 0.55 else "black", fontsize=10)
    return pd.DataFrame(cm, index=[CLASS_NAMES[i] for i in [1, 2, 3]], columns=[CLASS_NAMES[i] for i in [1, 2, 3]]).reset_index(names="true_class")


def _roc_pr_data(pred: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pred[pred["method"] == method].copy()
    roc_rows = []
    pr_rows = []
    summary_rows = []
    y_true = data["true_label"].to_numpy()
    for label in [1, 2, 3]:
        y_bin = (y_true == label).astype(int)
        score = data[f"prob_class_{label}"].to_numpy()
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, score)
        precision, recall, _ = precision_recall_curve(y_bin, score)
        auroc = roc_auc_score(y_bin, score)
        auprc = average_precision_score(y_bin, score)
        roc_rows.extend({"class_label": label, "class_name": CLASS_NAMES[label], "fpr": a, "tpr": b} for a, b in zip(fpr, tpr))
        pr_rows.extend({"class_label": label, "class_name": CLASS_NAMES[label], "recall": r, "precision": p} for p, r in zip(precision, recall))
        summary_rows.append({"class_label": label, "class_name": CLASS_NAMES[label], "auroc": auroc, "auprc": auprc})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), pd.DataFrame(summary_rows)


def create_all_figures(output_dir: Path, cfg: dict, metadata: pd.DataFrame, fold_metrics: pd.DataFrame, predictions: pd.DataFrame, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Create exactly eight final figure groups."""
    validations = []
    fs = (10.5, 6.8)
    title_size = cfg["plots"]["title_font_size"]
    label_size = cfg["plots"]["axis_font_size"]
    total_coefficients = int(cfg["features"]["total_coefficients"])
    num_qubits = int(cfg["quantum"]["num_qubits"])
    rounds = int(cfg["quantum"]["reuploading_rounds"])
    gate = str(cfg["quantum"]["encoding_gate"]).replace("_", "+")
    profiles = allocation_profiles(total_coefficients)
    measurements = num_qubits + (num_qubits * (num_qubits - 1)) // 2
    primary_method = _primary_method(cfg, fold_metrics)
    feature_top = cfg.get("plots", {}).get("workflow_feature_top", "Global\nDCT")
    feature_bottom = cfg.get("plots", {}).get("workflow_feature_bottom", "Local\nDCT")
    feature_node = cfg.get("plots", {}).get("workflow_feature_node", "Global and local DCT")
    for number, name in enumerate(FIGURE_NAMES, start=1):
        fig, axes = plt.subplots(2, 2, figsize=fs, constrained_layout=True)
        axes = axes.ravel()
        for ax in axes:
            ax.set_prop_cycle(color=PALETTE)
        if number == 1:
            examples = _example_samples(data_dir)
            counts = metadata["class_label"].value_counts().sort_index()
            patient_counts = metadata.groupby("class_label")["patient_id"].nunique().reset_index(name="patient_count")
            size_distribution = metadata.get("tumor_area_ratio", pd.Series(dtype=float)).dropna().to_frame("tumor_area_ratio")
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "slice_counts": counts.rename_axis("class_label").reset_index(name="slice_count"),
                    "patient_counts": patient_counts,
                    "tumor_area_ratios": size_distribution,
                },
            )
            if examples:
                for ax, sample in zip(axes[:3], examples):
                    _overlay_mask(ax, sample, f"{CLASS_NAMES[int(sample['label'])]} example")
            else:
                for ax in axes[:3]:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "Dataset unavailable", ha="center", va="center")
            ax = axes[3]
            x = np.arange(len(counts))
            width = 0.38
            ax.bar(x - width / 2, counts.values, width, label="Slices", color=PALETTE[0])
            ax.bar(x + width / 2, patient_counts["patient_count"], width, label="Patients", color=PALETTE[2])
            ax.set_xticks(x, [CLASS_NAMES[int(i)] for i in counts.index], rotation=15, ha="right")
            ax.set_title("Class Distribution")
            ax.set_ylabel("Count")
            ax.legend(frameon=False)
        elif number == 2:
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "workflow_nodes": pd.DataFrame(
                        {
                            "step_order": [1, 2, 3, 4, 5],
                            "node": ["Full MRI / tumour mask", feature_node, "Lesion-size allocation", f"{num_qubits}-qubit encoder", "Classifier"],
                        }
                    )
                },
            )
            for ax in axes:
                ax.axis("off")
            ax = axes[0]
            box_font = max(10, label_size - 3)
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 7)
            nodes = [
                (1.4, 5.5, "Full\nMRI", PALETTE[0]),
                (1.4, 3.3, "Tumour\nmask", PALETTE[2]),
                (4.4, 5.5, feature_top, PALETTE[0]),
                (4.4, 3.3, feature_bottom, PALETTE[2]),
                (8.0, 4.4, "Size-aware\nallocation", PALETTE[1]),
                (12.0, 4.4, f"{total_coefficients} fixed\nslots", PALETTE[3]),
            ]
            for x, y, text, color in nodes:
                ax.text(x, y, text, ha="center", va="center", fontsize=box_font, bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "white", "alpha": 0.9}, color="white")
            arrows = [((2.2, 5.5), (3.4, 5.5)), ((2.2, 3.3), (3.4, 3.3)), ((5.2, 5.5), (6.8, 4.7)), ((5.2, 3.3), (6.8, 4.1)), ((9.2, 4.4), (10.8, 4.4))]
            for start, end in arrows:
                ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#333333"})
            ax.set_title("Global-Local Feature Construction")
            ax = axes[1]
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 7)
            nodes = [
                (1.6, 4.6, "Angles", PALETTE[3]),
                (5.0, 4.6, f"{num_qubits}-qubit\n{gate} map", PALETTE[4]),
                (8.8, 4.6, f"{measurements} Z / ZZ\nfeatures", PALETTE[5]),
                (12.3, 4.6, "Classifier", PALETTE[6]),
            ]
            for x, y, text, color in nodes:
                ax.text(x, y, text, ha="center", va="center", fontsize=box_font, bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "white", "alpha": 0.9})
            for start, end in [((2.5, 4.6), (3.7, 4.6)), ((6.3, 4.6), (7.5, 4.6)), ((10.0, 4.6), (11.1, 4.6))]:
                ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#333333"})
            ax.set_title("Fixed Quantum Feature Map")
            axes[2].axis("off")
            axes[2].text(0.04, 0.78, "Fixed budget", fontsize=title_size, weight="bold")
            axes[2].text(
                0.04,
                0.5,
                f"Every quantum method uses {total_coefficients} input slots,\n{num_qubits} qubits, {rounds} encoding rounds,\nand {measurements} measurement features.",
                fontsize=label_size,
            )
            axes[3].axis("off")
            axes[3].text(0.04, 0.78, "Evaluation", fontsize=title_size, weight="bold")
            axes[3].text(0.04, 0.5, "Patient-disjoint folds prevent leakage.\nPatient-level metrics are emphasized.", fontsize=label_size)
        elif number in {3, 6}:
            metric = "patient_macro_f1" if "patient_macro_f1" in fold_metrics else "slice_macro_f1"
            crosstab = pd.crosstab(metadata["class_label"], metadata["lesion_size_category"]).reset_index() if "lesion_size_category" in metadata else pd.DataFrame()
            method_summary = fold_metrics.groupby("method", as_index=False)[metric].mean() if metric in fold_metrics else pd.DataFrame()
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "class_by_lesion_size": crosstab,
                    "method_macro_f1": method_summary,
                    "tumor_area_ratios": metadata.get("tumor_area_ratio", pd.Series(dtype=float)).dropna().to_frame("tumor_area_ratio"),
                    "allocation_mapping": pd.DataFrame(
                        {
                            "lesion_size": ["small", "medium", "large"],
                            "global_coefficients": [profiles["small"][0], profiles["medium"][0], profiles["large"][0]],
                            "local_coefficients": [profiles["small"][1], profiles["medium"][1], profiles["large"][1]],
                        }
                    ),
                },
            )
            if "lesion_size_category" in metadata:
                ct = pd.crosstab(metadata["class_label"].map(CLASS_NAMES), metadata["lesion_size_category"])
                ct[["small", "medium", "large"]].plot(kind="bar", ax=axes[0], color=PALETTE[:3])
            axes[0].set_title("Class by Lesion Size")
            axes[0].set_xlabel("")
            axes[0].tick_params(axis="x", rotation=15)
            _plot_barh(axes[1], _metric_summary(fold_metrics, metric).tail(10), "Mean Patient Macro F1" if number == 3 else "Allocation Ablation Macro F1", "Macro F1")
            axes[2].axis("off")
            axes[2].text(0.05, 0.84, "Allocation Rule", fontsize=title_size, weight="bold")
            axes[2].text(
                0.05,
                0.52,
                "Small lesion: "
                f"{profiles['small'][0]} global + {profiles['small'][1]} local\n"
                "Medium lesion: "
                f"{profiles['medium'][0]} global + {profiles['medium'][1]} local\n"
                "Large lesion: "
                f"{profiles['large'][0]} global + {profiles['large'][1]} local",
                fontsize=label_size,
                linespacing=1.35,
            )
            axes[3].hist(metadata.get("tumor_area_ratio", pd.Series([0])).dropna(), bins=30, color=PALETTE[1], alpha=0.85)
            axes[3].set_title("Tumour-Area Ratio Distribution")
            axes[3].set_xlabel("Tumour pixels / image pixels")
        elif number == 4:
            metric_names = ["patient_macro_f1", "patient_balanced_accuracy", "patient_macro_auroc", "patient_macro_auprc"]
            summaries = [_metric_summary(fold_metrics, metric).assign(metric=metric) for metric in metric_names if metric in fold_metrics]
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "fold_metrics": fold_metrics[["seed", "fold", "method", *[m for m in metric_names if m in fold_metrics]]],
                    "method_metric_summary": pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
                },
            )
            titles = ["Patient Macro F1", "Patient Balanced Accuracy", "Patient Macro AUROC", "Patient Macro AUPRC"]
            for i, metric in enumerate(metric_names):
                if metric in fold_metrics:
                    _plot_barh(axes[i], _metric_summary(fold_metrics, metric), titles[i], titles[i].split()[-1])
        elif number == 5:
            proposed = predictions[predictions["method"] == primary_method].copy()
            best = _best_baseline(fold_metrics, primary_method)
            baseline = predictions[predictions["method"] == best].copy()
            per_class_cols = [c for c in fold_metrics.columns if c.startswith("patient_class_") and (c.endswith("_recall") or c.endswith("_f1"))]
            proposed_cm = _plot_confusion(axes[0], predictions, primary_method, f"{METHOD_NAMES.get(primary_method, primary_method)} Patient Confusion")
            if best == primary_method:
                axes[1].axis("off")
                axes[1].text(0.04, 0.55, "No separate baseline predictions\nwere provided in this result set.", fontsize=label_size, va="center")
                baseline_cm = pd.DataFrame()
            else:
                baseline_cm = _plot_confusion(axes[1], predictions, best, f"{METHOD_NAMES.get(best, best)} Patient Confusion")
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "proposed_predictions": proposed,
                    "baseline_predictions": baseline,
                    "proposed_confusion_matrix": proposed_cm,
                    "baseline_confusion_matrix": baseline_cm,
                    "per_class_fold_metrics": fold_metrics[["seed", "fold", "method", *per_class_cols]],
                },
            )
            recall_cols = [f"patient_class_{i}_recall" for i in [1, 2, 3] if f"patient_class_{i}_recall" in fold_metrics]
            f1_cols = [f"patient_class_{i}_f1" for i in [1, 2, 3] if f"patient_class_{i}_f1" in fold_metrics]
            for ax, cols, title in [(axes[2], recall_cols, "Per-Class Recall"), (axes[3], f1_cols, "Per-Class F1")]:
                rows = []
                methods_to_plot = [primary_method] if best == primary_method else [primary_method, best]
                for method in methods_to_plot:
                    means = fold_metrics[fold_metrics["method"] == method][cols].mean()
                    for col, val in means.items():
                        cls = int(col.split("_")[2])
                        rows.append({"method": METHOD_NAMES.get(method, method), "class": CLASS_NAMES[cls], "value": val})
                df = pd.DataFrame(rows)
                if df.empty:
                    ax.axis("off")
                    ax.text(0.04, 0.55, "Per-class metrics unavailable.", fontsize=label_size, va="center")
                    continue
                pivot = df.pivot(index="class", columns="method", values="value")
                pivot.plot(kind="bar", ax=ax, color=PALETTE[: len(pivot.columns)])
                ax.set_title(title)
                ax.set_ylim(0, 1)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=15)
                ax.legend(frameon=False)
        elif number == 7:
            counts = logical_resource_counts(rounds, num_qubits, str(cfg["quantum"]["encoding_gate"]))
            resource = pd.DataFrame(
                {
                    "quantity": ["Qubits", "RY gates", "RZ gates", "CNOT gates", "Measurements", "Nominal depth"],
                    "value": [counts["num_qubits"], counts["ry_gates"], counts["rz_gates"], counts["cnot_gates"], counts["measurements"], counts["nominal_depth"]],
                }
            )
            robustness_status = pd.DataFrame(
                {
                    "analysis": ["Shot robustness", "Effective noise", "ROI perturbation", "Context margin", "Coefficient budget"],
                    "status": ["Not run in saved outputs"] * 5,
                }
            )
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "robustness_status": robustness_status,
                    "logical_resources": resource,
                },
            )
            axes[0].axis("off")
            axes[0].text(0.02, 0.85, "Robustness experiments", fontsize=title_size, weight="bold")
            axes[0].text(0.02, 0.45, "Shot, noise, ROI, context-margin, and\ncoefficient-budget robustness were not\npresent in these saved outputs.\nNo curves are fabricated.", fontsize=label_size)
            axes[1].bar(resource["quantity"], resource["value"], color=PALETTE[: len(resource)])
            axes[1].set_title("Fixed Circuit Resources")
            axes[1].tick_params(axis="x", rotation=20)
            _plot_barh(axes[2], _metric_summary(fold_metrics, "patient_macro_f1").tail(8), "Observed Method Performance", "Macro F1")
            axes[3].axis("off")
            status_text = "\n".join(f"{row.analysis}: {row.status}" for row in robustness_status.itertuples())
            axes[3].text(0.02, 0.8, status_text, fontsize=label_size, va="top")
        else:
            method = primary_method
            roc_df, pr_df, rocpr_summary = _roc_pr_data(predictions, method)
            _write_plot_data(
                output_dir,
                number,
                name,
                {
                    "predictions": predictions,
                    "proposed_roc_curve": roc_df,
                    "proposed_pr_curve": pr_df,
                    "proposed_roc_pr_summary": rocpr_summary,
                    "method_auc_summary": fold_metrics[
                        ["seed", "fold", "method", *[c for c in ["patient_macro_auroc", "patient_macro_auprc"] if c in fold_metrics]]
                    ],
                },
            )
            for cls, group in roc_df.groupby("class_name"):
                axes[0].plot(group["fpr"], group["tpr"], linewidth=2.0, label=cls)
            axes[0].plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.2)
            axes[0].set_title(f"{METHOD_NAMES.get(method, method)} ROC Curves")
            axes[0].set_xlabel("False positive rate")
            axes[0].set_ylabel("True positive rate")
            axes[0].legend(frameon=False)
            for cls, group in pr_df.groupby("class_name"):
                axes[1].plot(group["recall"], group["precision"], linewidth=2.0, label=cls)
            axes[1].set_title(f"{METHOD_NAMES.get(method, method)} Precision-Recall")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].legend(frameon=False)
            auroc_summary = _metric_summary(fold_metrics, "patient_macro_auroc")
            auprc_summary = _metric_summary(fold_metrics, "patient_macro_auprc")
            _plot_barh(axes[2], auroc_summary, "Method Macro AUROC", "AUROC")
            _plot_barh(axes[3], auprc_summary, "Method Macro AUPRC", "AUPRC")
        for ax in axes:
            ax.set_title(ax.get_title(), fontsize=title_size)
            ax.tick_params(labelsize=cfg["plots"]["tick_font_size"])
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
        validations.append(_save_and_validate(fig, output_dir, number, name, cfg))
    val_df = pd.DataFrame(validations)
    rep = output_dir / "reproducibility"
    rep.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(rep / "plot_validation.csv", index=False)
    return val_df
