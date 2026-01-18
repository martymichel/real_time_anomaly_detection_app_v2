"""
Validation Visualizer - Generate comprehensive statistics and plots for model validation.

Creates detailed reports including:
- Score distributions (good vs defect)
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Threshold analysis
- Per-image score plots
- Statistical summaries
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] Matplotlib not available - visualizations will be skipped")


class ValidationVisualizer:
    """Generate comprehensive validation statistics and visualizations."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots and statistics (e.g., projects/my_project/results/val/)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for consistent plotting
        if MATPLOTLIB_AVAILABLE:
            sns.set_style("darkgrid")
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 10

    def generate_full_report(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        threshold: float,
        metrics: Dict[str, float],
        good_image_names: Optional[List[str]] = None,
        defect_image_names: Optional[List[str]] = None
    ):
        """
        Generate complete validation report with all plots and statistics.

        Args:
            good_scores: Anomaly scores for good test images
            defect_scores: Anomaly scores for defect test images
            threshold: Optimized threshold value
            metrics: Dictionary with accuracy, precision, recall, f1_score, false_positives, false_negatives
            good_image_names: Optional list of good image filenames
            defect_image_names: Optional list of defect image filenames
        """
        print(f"\n{'='*60}")
        print("GENERATING VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Good images: {len(good_scores)}")
        print(f"  Defect images: {len(defect_scores)}")
        print(f"  Threshold: {threshold:.4f}")

        # Save raw scores
        self._save_raw_scores(good_scores, defect_scores, good_image_names, defect_image_names)

        # Generate statistics summary
        stats = self._compute_statistics(good_scores, defect_scores, threshold, metrics)
        self._save_statistics_json(stats)

        if not MATPLOTLIB_AVAILABLE:
            print("[WARN] Matplotlib not available - skipping visualizations")
            return

        # Generate plots
        try:
            self._plot_score_distributions(good_scores, defect_scores, threshold)
            self._plot_score_histograms(good_scores, defect_scores, threshold)
            self._plot_box_plots(good_scores, defect_scores)
            self._plot_per_image_scores(good_scores, defect_scores, threshold, good_image_names, defect_image_names)
            self._plot_confusion_matrix(metrics)
            self._plot_threshold_analysis(good_scores, defect_scores, threshold)
            self._plot_roc_curve(good_scores, defect_scores, threshold)
            self._plot_precision_recall_curve(good_scores, defect_scores, threshold)

            print(f"\n[OK] Validation report complete! Saved to: {self.output_dir}")
            print(f"{'='*60}\n")
        except Exception as e:
            error_msg = f"[ERROR] Failed to generate plots: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()

            # Write error to log file for debugging
            error_log_path = self.output_dir / "plot_generation_error.log"
            with open(error_log_path, 'w') as f:
                f.write(f"Error during plot generation:\n")
                f.write(f"{error_msg}\n\n")
                f.write("Full traceback:\n")
                traceback.print_exc(file=f)
            print(f"[INFO] Error details saved to: {error_log_path}")

    def _save_raw_scores(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        good_image_names: Optional[List[str]],
        defect_image_names: Optional[List[str]]
    ):
        """Save raw anomaly scores to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "good_scores": {
                "values": [float(s) for s in good_scores],
                "count": len(good_scores),
                "image_names": good_image_names if good_image_names else [f"good_{i+1:03d}.png" for i in range(len(good_scores))]
            },
            "defect_scores": {
                "values": [float(s) for s in defect_scores],
                "count": len(defect_scores),
                "image_names": defect_image_names if defect_image_names else [f"defect_{i+1:03d}.png" for i in range(len(defect_scores))]
            }
        }

        output_path = self.output_dir / "raw_scores.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  [OK] Raw scores saved: {output_path.name}")

    def _compute_statistics(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        threshold: float,
        metrics: Dict[str, float]
    ) -> Dict:
        """Compute comprehensive statistics."""
        good_arr = np.array(good_scores)
        defect_arr = np.array(defect_scores)

        stats = {
            "timestamp": datetime.now().isoformat(),
            "threshold": float(threshold),
            "metrics": {
                "accuracy": float(metrics.get('accuracy', 0)),
                "precision": float(metrics.get('precision', 0)),
                "recall": float(metrics.get('recall', 0)),
                "f1_score": float(metrics.get('f1_score', 0)),
                "false_positives": int(metrics.get('false_positives', 0)),
                "false_negatives": int(metrics.get('false_negatives', 0)),
                "true_positives": int(metrics.get('true_positives', len(defect_scores))),
                "true_negatives": int(metrics.get('true_negatives', len(good_scores)))
            },
            "good_images": {
                "count": len(good_scores),
                "min": float(good_arr.min()),
                "max": float(good_arr.max()),
                "mean": float(good_arr.mean()),
                "median": float(np.median(good_arr)),
                "std": float(good_arr.std()),
                "percentiles": {
                    "p25": float(np.percentile(good_arr, 25)),
                    "p50": float(np.percentile(good_arr, 50)),
                    "p75": float(np.percentile(good_arr, 75)),
                    "p90": float(np.percentile(good_arr, 90)),
                    "p95": float(np.percentile(good_arr, 95)),
                    "p99": float(np.percentile(good_arr, 99))
                }
            },
            "defect_images": {
                "count": len(defect_scores),
                "min": float(defect_arr.min()),
                "max": float(defect_arr.max()),
                "mean": float(defect_arr.mean()),
                "median": float(np.median(defect_arr)),
                "std": float(defect_arr.std()),
                "percentiles": {
                    "p1": float(np.percentile(defect_arr, 1)),
                    "p5": float(np.percentile(defect_arr, 5)),
                    "p10": float(np.percentile(defect_arr, 10)),
                    "p25": float(np.percentile(defect_arr, 25)),
                    "p50": float(np.percentile(defect_arr, 50)),
                    "p75": float(np.percentile(defect_arr, 75))
                }
            },
            "separation": {
                "gap": float(defect_arr.min() - good_arr.max()),
                "overlap": self._compute_overlap(good_arr, defect_arr),
                "separation_quality": self._compute_separation_quality(good_arr, defect_arr)
            }
        }

        return stats

    def _compute_overlap(self, good_scores: np.ndarray, defect_scores: np.ndarray) -> float:
        """Compute overlap between good and defect score distributions."""
        good_max = good_scores.max()
        defect_min = defect_scores.min()

        if good_max < defect_min:
            return 0.0  # No overlap

        # Count overlapping scores
        overlap_count = np.sum(good_scores > defect_min) + np.sum(defect_scores < good_max)
        total_count = len(good_scores) + len(defect_scores)

        return float(overlap_count / total_count)

    def _compute_separation_quality(self, good_scores: np.ndarray, defect_scores: np.ndarray) -> str:
        """Assess separation quality between good and defect distributions."""
        gap = defect_scores.min() - good_scores.max()

        if gap > 0.1:
            return "excellent"
        elif gap > 0.05:
            return "good"
        elif gap > 0:
            return "fair"
        else:
            return "poor"

    def _save_statistics_json(self, stats: Dict):
        """Save statistics to JSON file."""
        output_path = self.output_dir / "statistics.json"
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  [OK] Statistics saved: {output_path.name}")

    def _plot_score_distributions(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        threshold: float
    ):
        """Plot score distributions with KDE."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # KDE plots
        sns.kdeplot(data=good_scores, label='Good (OK)', color='green', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(data=defect_scores, label='Defect (NOK)', color='red', fill=True, alpha=0.3, ax=ax)

        # Threshold line
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution (Good vs Defect)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "score_distribution_kde.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_score_histograms(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        threshold: float
    ):
        """Plot score histograms."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Histograms
        bins = np.linspace(
            min(min(good_scores), min(defect_scores)),
            max(max(good_scores), max(defect_scores)),
            30
        )

        ax.hist(good_scores, bins=bins, alpha=0.5, color='green', label='Good (OK)', edgecolor='black')
        ax.hist(defect_scores, bins=bins, alpha=0.5, color='red', label='Defect (NOK)', edgecolor='black')

        # Threshold line
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Score Histogram (Good vs Defect)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "score_histogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_box_plots(
        self,
        good_scores: List[float],
        defect_scores: List[float]
    ):
        """Plot box plots for score comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        data = [good_scores, defect_scores]
        labels = ['Good (OK)', 'Defect (NOK)']
        colors = ['green', 'red']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title('Score Distribution Box Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / "score_boxplot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_per_image_scores(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        threshold: float,
        good_image_names: Optional[List[str]],
        defect_image_names: Optional[List[str]]
    ):
        """Plot per-image scores."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Good images
        x_good = range(len(good_scores))
        ax1.scatter(x_good, good_scores, color='green', s=100, alpha=0.6, edgecolors='black', linewidth=1)
        ax1.axhline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax1.set_xlabel('Image Index', fontsize=12)
        ax1.set_ylabel('Anomaly Score', fontsize=12)
        ax1.set_title('Good Images - Per-Image Scores', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Defect images
        x_defect = range(len(defect_scores))
        ax2.scatter(x_defect, defect_scores, color='red', s=100, alpha=0.6, edgecolors='black', linewidth=1)
        ax2.axhline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax2.set_xlabel('Image Index', fontsize=12)
        ax2.set_ylabel('Anomaly Score', fontsize=12)
        ax2.set_title('Defect Images - Per-Image Scores', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "per_image_scores.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_confusion_matrix(self, metrics: Dict[str, float]):
        """Plot confusion matrix."""
        tn = int(metrics.get('true_negatives', 0))
        fp = int(metrics.get('false_positives', 0))
        fn = int(metrics.get('false_negatives', 0))
        tp = int(metrics.get('true_positives', 0))

        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted Good', 'Predicted Defect'],
            yticklabels=['Actual Good', 'Actual Defect'],
            ax=ax, annot_kws={'size': 16, 'weight': 'bold'}
        )

        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual Class', fontsize=12)
        ax.set_xlabel('Predicted Class', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_threshold_analysis(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        optimal_threshold: float
    ):
        """Plot threshold analysis showing metrics vs threshold."""
        # Generate threshold range
        all_scores = sorted(good_scores + defect_scores)
        thresholds = np.linspace(all_scores[0], all_scores[-1], 100)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for thresh in thresholds:
            tp = sum(s > thresh for s in defect_scores)
            tn = sum(s <= thresh for s in good_scores)
            fp = sum(s > thresh for s in good_scores)
            fn = sum(s <= thresh for s in defect_scores)

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)

        # Mark optimal threshold
        ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.4f}')

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        output_path = self.output_dir / "threshold_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_roc_curve(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        optimal_threshold: float
    ):
        """Plot ROC curve."""
        # Generate threshold range
        all_scores = sorted(good_scores + defect_scores)
        thresholds = np.linspace(all_scores[0], all_scores[-1], 200)

        tprs = []
        fprs = []

        for thresh in thresholds:
            tp = sum(s > thresh for s in defect_scores)
            fn = sum(s <= thresh for s in defect_scores)
            fp = sum(s > thresh for s in good_scores)
            tn = sum(s <= thresh for s in good_scores)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tprs.append(tpr)
            fprs.append(fpr)

        # Calculate AUC
        auc = np.trapz(tprs, fprs)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fprs, tprs, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

        # Mark optimal threshold point
        tp = sum(s > optimal_threshold for s in defect_scores)
        fn = sum(s <= optimal_threshold for s in defect_scores)
        fp = sum(s > optimal_threshold for s in good_scores)
        tn = sum(s <= optimal_threshold for s in good_scores)

        optimal_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        optimal_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        ax.scatter([optimal_fpr], [optimal_tpr], color='red', s=200, marker='*',
                  label=f'Optimal (FPR={optimal_fpr:.3f}, TPR={optimal_tpr:.3f})', zorder=5)

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")

    def _plot_precision_recall_curve(
        self,
        good_scores: List[float],
        defect_scores: List[float],
        optimal_threshold: float
    ):
        """Plot Precision-Recall curve."""
        # Generate threshold range
        all_scores = sorted(good_scores + defect_scores)
        thresholds = np.linspace(all_scores[0], all_scores[-1], 200)

        precisions = []
        recalls = []

        for thresh in thresholds:
            tp = sum(s > thresh for s in defect_scores)
            fn = sum(s <= thresh for s in defect_scores)
            fp = sum(s > thresh for s in good_scores)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recalls, precisions, linewidth=2, label='Precision-Recall Curve')

        # Mark optimal threshold point
        tp = sum(s > optimal_threshold for s in defect_scores)
        fn = sum(s <= optimal_threshold for s in defect_scores)
        fp = sum(s > optimal_threshold for s in good_scores)

        optimal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        optimal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax.scatter([optimal_recall], [optimal_precision], color='red', s=200, marker='*',
                  label=f'Optimal (Recall={optimal_recall:.3f}, Prec={optimal_precision:.3f})', zorder=5)

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        output_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {output_path.name}")
