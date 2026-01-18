"""
Project Manager for Anomaly Detection
======================================

Manages project structure, configuration, and data collection workflow.

Project Structure:
    projects/
    └── project_name/
        ├── config.json              # Project configuration
        ├── memory_bank.pt           # Saved memory bank (PyTorch tensor)
        ├── images/
        │   ├── train/
        │   │   └── good/            # 30 training images (OK)
        │   └── test/
        │       ├── good/            # 20 test images (OK)
        │       └── defect/          # 20 test images (NOK)
        └── results/
            ├── threshold_validation.json
            └── reference_image.png
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
import torch
import cv2
from PIL import Image


class ProjectConfig:
    """Project configuration data structure."""

    def __init__(
        self,
        project_name: str,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",  # DINOv3 ViT-L/16
        shots: int = 30,
        knn_k: int = 5,
        metric: str = "cosine",
        image_size: int = 512,  # Single training resolution (must be multiple of 16)
        batch_size: int = 4,
        train_target: int = 16,
        test_good_target: int = 10,
        test_defect_target: int = 10,
        coreset_percentage: Optional[float] = None,  # 1-100% or None (disabled)
        coreset_method: str = "random",  # Coreset method: random, stratified, fps_gpu, importance, greedy
        selected_layers: Optional[List[int]] = None,  # Which DINOv3 layers to extract (None = last layer only)
        # Motion Detection Parameters (FPS-adaptive)
        enable_motion_filter: bool = False,
        motion_high_threshold: float = 0.05,  # 5% pixel motion to trigger
        motion_low_threshold: float = 0.01,   # 1% pixel motion to release
        motion_stabilization_time: float = 0.5,  # seconds to wait after motion stops (default: 0.5s)
        motion_learning_time: float = 10.0,      # seconds for background learning
    ):
        self.project_name = project_name
        self.model_name = model_name
        self.shots = shots
        self.knn_k = knn_k
        self.metric = metric
        self.image_size = image_size  # Single training resolution
        self.batch_size = batch_size
        self.train_target = train_target
        self.test_good_target = test_good_target
        self.test_defect_target = test_defect_target
        self.coreset_percentage = coreset_percentage  # Optional coreset reduction
        self.coreset_method = coreset_method  # Coreset selection method
        self.selected_layers = selected_layers if selected_layers is not None else [-1]  # Default: last layer only
        # Motion Detection
        self.enable_motion_filter = enable_motion_filter
        self.motion_high_threshold = motion_high_threshold
        self.motion_low_threshold = motion_low_threshold
        self.motion_stabilization_time = motion_stabilization_time
        self.motion_learning_time = motion_learning_time
        # Other fields
        self.threshold: Optional[float] = None
        self.created = datetime.now().isoformat()
        self.train_images_count = 0
        self.test_good_count = 0
        self.test_defect_count = 0
        self.validation_results: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "project_name": self.project_name,
            "created": self.created,
            "model_name": self.model_name,
            "shots": self.shots,
            "knn_k": self.knn_k,
            "metric": self.metric,
            "threshold": self.threshold,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "train_target": self.train_target,
            "test_good_target": self.test_good_target,
            "test_defect_target": self.test_defect_target,
            "coreset_percentage": self.coreset_percentage,
            "coreset_method": self.coreset_method,
            "selected_layers": self.selected_layers,
            "enable_motion_filter": self.enable_motion_filter,
            "motion_high_threshold": self.motion_high_threshold,
            "motion_low_threshold": self.motion_low_threshold,
            "motion_stabilization_time": self.motion_stabilization_time,
            "motion_learning_time": self.motion_learning_time,
            "train_images_count": self.train_images_count,
            "test_good_count": self.test_good_count,
            "test_defect_count": self.test_defect_count,
            "validation_results": self.validation_results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        """Create from dictionary."""
        config = cls(
            project_name=data["project_name"],
            model_name=data.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m"),  # DINOv3 default
            shots=data.get("shots", 30),
            knn_k=data.get("knn_k", 5),
            metric=data.get("metric", "cosine"),
            image_size=data.get("image_size", 512),
            batch_size=data.get("batch_size", 4),
            train_target=data.get("train_target", 16),
            test_good_target=data.get("test_good_target", 10),
            test_defect_target=data.get("test_defect_target", 10),
            coreset_percentage=data.get("coreset_percentage"),  # Optional
            coreset_method=data.get("coreset_method", "random"),  # Default: random
            selected_layers=data.get("selected_layers", [-1]),  # Default: last layer only
            # Motion Detection (with defaults for backward compatibility)
            enable_motion_filter=data.get("enable_motion_filter", False),
            motion_high_threshold=data.get("motion_high_threshold", 0.05),
            motion_low_threshold=data.get("motion_low_threshold", 0.01),
            motion_stabilization_time=data.get("motion_stabilization_time", 0.5),
            motion_learning_time=data.get("motion_learning_time", 10.0),
        )
        config.created = data.get("created", datetime.now().isoformat())
        config.threshold = data.get("threshold")
        config.train_images_count = data.get("train_images_count", 0)
        config.test_good_count = data.get("test_good_count", 0)
        config.test_defect_count = data.get("test_defect_count", 0)
        config.validation_results = data.get("validation_results")
        return config


class ProjectManager:
    """Manages anomaly detection projects."""

    def __init__(self, projects_root: Path = None):
        """
        Initialize project manager.

        Args:
            projects_root: Root directory for all projects
        """
        if projects_root is None:
            # Default: projects/ folder next to this script
            projects_root = Path(__file__).parent / "projects"

        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(parents=True, exist_ok=True)

        self.current_project: Optional[str] = None
        self.current_config: Optional[ProjectConfig] = None

    def list_projects(self) -> List[str]:
        """List all available projects."""
        projects = []
        for p in self.projects_root.iterdir():
            if p.is_dir() and (p / "config.json").exists():
                projects.append(p.name)
        return sorted(projects)

    def project_exists(self, project_name: str) -> bool:
        """Check if project exists."""
        return (self.projects_root / project_name / "config.json").exists()

    def get_project_path(self, project_name: str) -> Path:
        """Get path to project directory."""
        return self.projects_root / project_name

    def create_project(self, config: ProjectConfig) -> Path:
        """
        Create new project with directory structure.

        Args:
            config: Project configuration

        Returns:
            Path to project directory
        """
        project_path = self.get_project_path(config.project_name)

        if project_path.exists():
            raise ValueError(f"Project '{config.project_name}' already exists!")

        # Create directory structure
        (project_path / "images" / "background").mkdir(parents=True, exist_ok=True)  # Background baseline images
        (project_path / "images" / "train" / "good").mkdir(parents=True, exist_ok=True)
        (project_path / "images" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (project_path / "images" / "test" / "defect").mkdir(parents=True, exist_ok=True)
        (project_path / "results").mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config(project_path, config)

        print(f"[OK] Project created: {project_path}")

        self.current_project = config.project_name
        self.current_config = config

        return project_path

    def load_project(self, project_name: str) -> ProjectConfig:
        """
        Load existing project.

        Args:
            project_name: Name of project to load

        Returns:
            Project configuration
        """
        project_path = self.get_project_path(project_name)
        config_file = project_path / "config.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Project '{project_name}' not found!")

        with open(config_file, 'r') as f:
            data = json.load(f)

        config = ProjectConfig.from_dict(data)

        self.current_project = project_name
        self.current_config = config

        print(f"[OK] Project loaded: {project_name}")
        print(f"  Created: {config.created}")
        print(f"  Model: {config.model_name}")
        print(f"  Threshold: {config.threshold}")

        return config

    def save_config(self):
        """Save current project configuration."""
        if self.current_project is None or self.current_config is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        self._save_config(project_path, self.current_config)

    def _save_config(self, project_path: Path, config: ProjectConfig):
        """Internal: Save config to file."""
        config_file = project_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def save_image(
        self,
        image: np.ndarray,
        category: str,
        index: int
    ) -> Path:
        """
        Save captured image to project.

        Args:
            image: Image as numpy array [H, W, 3]
            category: 'background', 'train/good', 'test/good', or 'test/defect'
            index: Image index (1-based)

        Returns:
            Path where image was saved
        """
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        image_path = project_path / "images" / category / f"{index:03d}.png"

        # Ensure directory exists
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        pil_img = Image.fromarray(image)
        pil_img.save(image_path)

        if self.current_config is None:
            raise RuntimeError("No project configuration loaded!")

        # Update counts
        if category == "train/good":
            self.current_config.train_images_count += 1
        elif category == "test/good":
            self.current_config.test_good_count += 1
        elif category == "test/defect":
            self.current_config.test_defect_count += 1
        # Note: background images are not counted in config

        return image_path

    def save_reference_image(self, image: np.ndarray):
        """Save a reference image for project preview."""
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        ref_path = project_path / "results" / "reference_image.png"

        pil_img = Image.fromarray(image)
        pil_img.save(ref_path)

    def get_reference_image_path(self) -> Optional[Path]:
        """Get path to reference image if it exists."""
        if self.current_project is None:
            return None

        project_path = self.get_project_path(self.current_project)
        ref_path = project_path / "results" / "reference_image.png"

        return ref_path if ref_path.exists() else None

    def save_memory_bank(self, memory_bank: torch.Tensor):
        """
        Save memory bank tensor to project.

        Args:
            memory_bank: Memory bank tensor
        """
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        mb_path = project_path / "memory_bank.pt"

        torch.save(memory_bank, mb_path)
        print(f"[OK] Memory bank saved: {mb_path}")

    def load_memory_bank(self):
        """
        Load memory bank from project.

        Returns:
            memory_bank tensor or None if file not found
        """
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        mb_path = project_path / "memory_bank.pt"

        if not mb_path.exists():
            return None

        memory_bank = torch.load(mb_path)
        print(f"[OK] Memory bank loaded: {memory_bank.shape}")
        return memory_bank

    def save_validation_results(
        self,
        threshold: float,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        false_positives: int = 0,
        false_negatives: int = 0,
        true_positives: int = 0,
        true_negatives: int = 0
    ):
        """
        Save threshold validation results.

        Args:
            threshold: Optimal threshold
            accuracy: Classification accuracy
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            false_positives: Number of false positives (good flagged as defect)
            false_negatives: Number of false negatives (defect flagged as good)
            true_positives: Number of true positives (defect correctly detected)
            true_negatives: Number of true negatives (good correctly detected)
        """
        if self.current_project is None or self.current_config is None:
            raise RuntimeError("No project loaded!")

        # Convert all NumPy floats to native Python floats for JSON compatibility
        threshold = float(threshold)
        accuracy = float(accuracy)
        precision = float(precision)
        recall = float(recall)
        f1_score = float(f1_score)
        false_positives = int(false_positives)
        false_negatives = int(false_negatives)
        true_positives = int(true_positives)
        true_negatives = int(true_negatives)

        # Update config
        self.current_config.threshold = threshold
        self.current_config.validation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "true_negatives": true_negatives
        }

        # Save detailed results
        project_path = self.get_project_path(self.current_project)
        results_file = project_path / "results" / "threshold_validation.json"

        results = {
            "threshold": threshold,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "true_negatives": true_negatives
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save config
        self.save_config()

        print(f"[OK] Validation results saved")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1-Score:  {f1_score:.2%}")
        print(f"  True Positives:  {true_positives}")
        print(f"  True Negatives:  {true_negatives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")

    def get_background_images_path(self) -> Path:
        """Get path to background baseline images."""
        if self.current_project is None:
            raise RuntimeError("No project loaded!")
        return self.get_project_path(self.current_project) / "images" / "background"

    def get_train_images_path(self) -> Path:
        """Get path to training images."""
        if self.current_project is None:
            raise RuntimeError("No project loaded!")
        return self.get_project_path(self.current_project) / "images" / "train"

    def get_test_images_paths(self) -> Tuple[Path, Path]:
        """Get paths to test images (good, defect)."""
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        test_path = self.get_project_path(self.current_project) / "images" / "test"
        return test_path / "good", test_path / "defect"

    def get_project_summary(self) -> Dict[str, Any]:
        """Get summary of current project."""
        if self.current_config is None:
            raise RuntimeError("No project loaded!")

        return {
            "name": self.current_config.project_name,
            "created": self.current_config.created,
            "model": self.current_config.model_name.split("/")[-1],
            "images": {
                "train_good": self.current_config.train_images_count,
                "test_good": self.current_config.test_good_count,
                "test_defect": self.current_config.test_defect_count,
                "total": (
                    self.current_config.train_images_count +
                    self.current_config.test_good_count +
                    self.current_config.test_defect_count
                )
            },
            "threshold": self.current_config.threshold,
            "validation": self.current_config.validation_results,
            "ready": self.is_ready_for_inference()
        }

    def is_ready_for_inference(self) -> bool:
        """Check if project is ready for live inference."""
        if self.current_config is None:
            return False

        # Check if memory bank exists
        if self.current_project:
            mb_path = self.get_project_path(self.current_project) / "memory_bank.pt"
            if not mb_path.exists():
                return False

        # Check if threshold is set
        if self.current_config.threshold is None:
            return False

        # Check if we have enough training images
        if self.current_config.train_images_count < self.current_config.shots:
            return False

        return True

    def has_training_images(self) -> bool:
        """Check if project has training images (for resume training)."""
        if self.current_project is None:
            return False

        train_path = self.get_train_images_path() / "good"
        test_good_path, test_defect_path = self.get_test_images_paths()

        # Count actual image files
        train_count = len(list(train_path.glob("*.png"))) if train_path.exists() else 0
        test_good_count = len(list(test_good_path.glob("*.png"))) if test_good_path.exists() else 0
        test_defect_count = len(list(test_defect_path.glob("*.png"))) if test_defect_path.exists() else 0

        # Need at least minimum training images
        return train_count >= 10 and test_good_count >= 5 and test_defect_count >= 5

    def get_image_path(self, category: str, index: int) -> Path:
        """
        Get path to a specific image.

        Args:
            category: 'train/good', 'test/good', or 'test/defect'
            index: Image index (1-based)

        Returns:
            Path to image file
        """
        if self.current_project is None:
            raise RuntimeError("No project loaded!")

        project_path = self.get_project_path(self.current_project)
        return project_path / "images" / category / f"{index:03d}.png"

    def delete_project(self, project_name: str):
        """
        Delete a project completely.

        Args:
            project_name: Name of project to delete

        Raises:
            FileNotFoundError: If project doesn't exist
        """
        project_path = self.get_project_path(project_name)

        if not project_path.exists():
            raise FileNotFoundError(f"Project '{project_name}' not found!")

        # Remove entire project directory
        shutil.rmtree(project_path)

        # Clear current project if it was deleted
        if self.current_project == project_name:
            self.current_project = None
            self.current_config = None

        print(f"[OK] Project '{project_name}' deleted completely")
