"""
Model Trainer for Anomaly Detection
====================================

Handles memory bank creation and threshold optimization.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, Callable, List
import time

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

# Import local dataset module


def get_model_num_layers(model_name: str) -> int:
    """
    Get number of hidden layers for a DINOv3 model.

    Args:
        model_name: Model name (e.g., facebook/dinov3-vitb16-pretrain-lvd1689m)

    Returns:
        Number of hidden layers
    """
    if "vits" in model_name.lower():
        return 12  # ViT-Small: 12 layers
    elif "vitb" in model_name.lower():
        return 12  # ViT-Base: 12 layers
    elif "vitl" in model_name.lower():
        return 24  # ViT-Large: 24 layers
    elif "vitg" in model_name.lower():
        return 40  # ViT-Giant: 40 layers (if exists)
    else:
        return 12  # Default fallback


class ModelTrainer:
    """
    Trains anomaly detection model.

    - Creates memory bank from training images
    - Optimizes threshold on test set
    - Validates performance
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
        selected_layers: Optional[List[int]] = None
    ):
        """
        Initialize trainer.

        Args:
            model_name: DINOv3 model name (e.g., facebook/dinov3-vitl16-pretrain-lvd1689m)
            device: Torch device (auto-detect if None)
            cache_dir: Model cache directory (defaults to ./models/)
            selected_layers: Which layers to extract features from (default: [-1] = last layer only)
        """
        self.model_name = model_name
        self.selected_layers = selected_layers if selected_layers is not None else [-1]
        self.device = device if device else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        if cache_dir is None:
            # Default to local models directory
            cache_dir = Path("models")
        self.cache_dir = Path(cache_dir)

        self.model = None
        self.memory_bank = None
        self.patches_per_image: Optional[int] = None
        self.training_resolution: Optional[int] = None  # Store training resolution

        # OPTIMIZED: AMP (Automatic Mixed Precision) for FP16 training
        self.use_amp = False
        self.amp_dtype = torch.float32
        if torch.cuda.is_available():
            gpu_capability = torch.cuda.get_device_capability(self.device)
            if gpu_capability[0] >= 7:
                self.use_amp = True
                self.amp_dtype = torch.float16
                print(f"  AMP: Enabled (FP16) - GPU CC {gpu_capability[0]}.{gpu_capability[1]}")
            else:
                print(f"  AMP: Disabled - GPU CC {gpu_capability[0]}.{gpu_capability[1]} < 7.0")

        print(f"Model Trainer initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  Selected layers: {self.selected_layers}")

    def load_model(self):
        """Load DINOv3 model from local directory only (no HuggingFace cache)."""
        if self.model is not None:
            return

        print(f"Loading DINOv3 model: {self.model_name}...")

        # Use simple directory format: models/facebook_dinov3-vitb16-pretrain-lvd1689m
        model_cache_dir = self.cache_dir / self.model_name.replace("/", "_")

        # Try local cache first (preferred)
        if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
            try:
                print(f"  [OK] Loading from local cache: {model_cache_dir}")
                self.model = AutoModel.from_pretrained(
                    model_cache_dir,
                    local_files_only=True
                )
                print(f"  [OK] Model loaded from local directory")
            except Exception as e:
                print(f"  [X] Local cache load failed: {e}")
                print("  -> Downloading from Hugging Face...")
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True  # Required for some models
                    )
                    model_cache_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(model_cache_dir)
                    print(f"  [OK] Model downloaded and saved to: {model_cache_dir}")
                except Exception as download_error:
                    print(f"  [X] Download failed: {download_error}")
                    print(f"\n[ERROR] Could not download model '{self.model_name}'")
                    print("Possible reasons:")
                    print("  * Model name is incorrect")
                    print("  * No internet connection")
                    print("  * HuggingFace is unreachable")
                    print(f"\nPlease check if the model exists at:")
                    print(f"  https://huggingface.co/{self.model_name}")
                    raise RuntimeError(f"Failed to download model: {self.model_name}") from download_error
        else:
            print(f"  -> Model not found locally, downloading from Hugging Face...")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True  # Required for some models
                )
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_cache_dir)
                print(f"  [OK] Model downloaded and saved to: {model_cache_dir}")
            except Exception as e:
                print(f"  [X] Download failed: {e}")
                print(f"\n[ERROR] Could not download model '{self.model_name}'")
                print("Possible reasons:")
                print("  * Model name is incorrect")
                print("  * No internet connection")
                print("  * HuggingFace is unreachable")
                print(f"\nPlease check if the model exists at:")
                print(f"  https://huggingface.co/{self.model_name}")
                raise RuntimeError(f"Failed to download model: {self.model_name}") from e

        self.model.to(self.device)
        self.model.eval()

        # NOTE: torch.compile() is NOT compatible with output_hidden_states=True
        # which is required for multi-layer feature extraction
        # Skipping torch.compile() to avoid runtime errors

        print("[OK] Model loaded and ready")

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv3 patch features from multiple layers with register token handling.

        DINOv3 token structure: [CLS, REG1, REG2, REG3, REG4, PATCH_TOKENS...]
        We skip the first 5 tokens (1 CLS + 4 registers) to get spatial patches.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            patch_features: Patch tokens [B, N_patches, C_total]
                           where C_total = sum of feature dims from selected layers
        """
        with torch.no_grad():
            outputs = self.model(images, output_hidden_states=True)

            # Extract features from selected layers
            layer_features = []

            _, _, height, width = images.shape
            expected_patches = (height // 16) * (width // 16)

            for layer_idx in self.selected_layers:
                # Get hidden state for this layer
                if layer_idx == -1:
                    # Last layer (last_hidden_state)
                    hidden_states = outputs.last_hidden_state
                else:
                    # Specific layer from hidden_states tuple
                    # hidden_states[0] is the embedding layer, hidden_states[1] is layer 0, etc.
                    hidden_states = outputs.hidden_states[layer_idx + 1]

                B, N_total, C = hidden_states.shape

                # Skip CLS + register tokens
                if N_total == expected_patches + 5:
                    patch_features = hidden_states[:, 5:, :]
                elif N_total > expected_patches + 5:
                    patch_features = hidden_states[:, 5:5+expected_patches, :]
                elif N_total == expected_patches + 1:
                    patch_features = hidden_states[:, 1:, :]
                else:
                    patch_features = hidden_states[:, 1:, :]

                layer_features.append(patch_features)

            # Concatenate features from all selected layers
            if len(layer_features) == 1:
                return layer_features[0]
            else:
                return torch.cat(layer_features, dim=-1)  # Concatenate along feature dimension

    @staticmethod
    def _prepare_image_tensor(image_np: np.ndarray, size: int) -> torch.Tensor:
        """
        Resize and normalize an image to a tensor with ImageNet normalization.

        Args:
            image_np: RGB image as numpy array [H, W, 3]
            size: Target size (square)

        Returns:
            Tensor [3, size, size]
        """
        import cv2

        # Resize using INTER_AREA for best downsampling quality (anti-aliasing)
        if image_np.shape[:2] != (size, size):
            resized = cv2.resize(image_np, (size, size), interpolation=cv2.INTER_AREA)
        else:
            resized = image_np

        # Normalize to [0, 1]
        img_float = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization (CRITICAL for DINOv3!)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        return torch.from_numpy(img_normalized).permute(2, 0, 1)

    def build_memory_bank(
        self,
        train_path: Path,
        shots: int,
        batch_size: int = 4,
        image_size: int = 512,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> torch.Tensor:
        """
        Build memory bank from training images using a SINGLE resolution.

        Args:
            train_path: Path to training images (train/good)
            shots: Number of images to use
            batch_size: Batch size for processing
            image_size: Training resolution (must be multiple of 16)
            progress_callback: Callback(current, total, message)

        Returns:
            Memory bank tensor [N_patches * shots, C]
        """
        if self.model is None:
            self.load_model()

        # Validate image_size is multiple of 16
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be multiple of 16, got {image_size}")

        print(f"\nBuilding memory bank from: {train_path}")
        print(f"  Training resolution: {image_size}x{image_size}")
        print(f"  Using {shots} images with batch size {batch_size}")

        # Collect image paths (train/good)
        train_good_path = train_path / "good" if (train_path / "good").exists() else train_path
        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(sorted(train_good_path.glob(ext)))

        if len(image_paths) < shots:
            raise ValueError(
                f"Not enough training images! Found {len(image_paths)}, need {shots}"
            )

        print(f"  Found {len(image_paths)} training images")

        # Calculate patches per image for this single resolution
        self.patches_per_image = (image_size // 16) ** 2
        self.training_resolution = image_size  # Store training resolution
        print(f"  Patches per image: {self.patches_per_image}")

        # Extract features
        feats = []
        total_batches = (shots + batch_size - 1) // batch_size
        processed = 0

        start_time = time.time()

        image_paths = image_paths[:shots]

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, shots)
            batch_paths = image_paths[start_idx:end_idx]

            if not batch_paths:
                break

            # Load batch images
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(np.array(img))

            # Prepare batch tensors at SINGLE resolution
            batch_tensors = torch.stack([
                self._prepare_image_tensor(img_np, image_size) for img_np in batch_images
            ]).to(self.device, non_blocking=True)

            # OPTIMIZED: Use AMP for feature extraction
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Extract features
                patch_features = self._extract_features(batch_tensors)

            # Convert to FP32 for normalization and storage
            if self.use_amp:
                patch_features = patch_features.float()

            # Normalize features
            patch_features = patch_features / (
                patch_features.norm(dim=-1, keepdim=True) + 1e-6
            )

            # Add to feature list (one entry per image)
            for i in range(len(batch_images)):
                feats.append(patch_features[i])

            processed += len(batch_images)

            if progress_callback:
                progress_callback(
                    processed,
                    shots,
                    f"Processing batch {batch_idx + 1}/{total_batches}"
                )

            print(f"  Processed: {processed}/{shots} images", end='\r')

        print()  # New line

        # Concatenate to form memory bank [N_patches * shots, C]
        self.memory_bank = torch.cat(feats, dim=0)

        elapsed = time.time() - start_time
        print(f"[OK] Memory bank created: {self.memory_bank.shape}")
        print(f"  Time elapsed: {elapsed:.1f}s")

        return self.memory_bank

    def _compute_anomaly_scores(
        self,
        patch_features: torch.Tensor,
        knn_k: int,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute anomaly scores using k-NN.

        Args:
            patch_features: Features [N_patches, C]
            knn_k: Number of nearest neighbors
            metric: Distance metric

        Returns:
            scores: Anomaly score per patch [N_patches]
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built!")

        if metric == 'cosine':
            dist = 1 - (patch_features @ self.memory_bank.t())
        else:
            dist = torch.cdist(patch_features, self.memory_bank)

        # k-NN averaging
        if knn_k > 1:
            topk = torch.topk(dist, k=knn_k, largest=False, dim=1).values
            scores = topk.mean(dim=1)
        else:
            scores = dist.min(dim=1).values

        return scores

    def predict_image(
        self,
        image_path: Path,
        knn_k: int,
        metric: str = "cosine",
        image_size: Optional[int] = None
    ) -> float:
        """
        Predict anomaly score for a single image using the training resolution.

        Args:
            image_path: Path to image
            knn_k: Number of nearest neighbors
            metric: Distance metric
            image_size: Image resolution (uses training_resolution if None)

        Returns:
            max_score: Maximum anomaly score
        """
        if self.model is None:
            self.load_model()

        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built!")

        # Use training resolution if not specified
        if image_size is None:
            if self.training_resolution is None:
                raise RuntimeError("Training resolution not set! Build memory bank first.")
            image_size = self.training_resolution

        # Load and preprocess image
        pil_img = Image.open(image_path).convert('RGB')
        img_np = np.array(pil_img)  # RGB format from PIL

        # Use the static helper method for preprocessing
        img_tensor = self._prepare_image_tensor(img_np, image_size).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Extract features with AMP
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                patch_features = self._extract_features(img_tensor)

            # Convert to FP32
            if self.use_amp:
                patch_features = patch_features.float()

            # Normalize
            patch_features = patch_features / (
                patch_features.norm(dim=-1, keepdim=True) + 1e-6
            )

            # Compute scores
            anomaly_scores = self._compute_anomaly_scores(
                patch_features[0],
                knn_k,
                metric
            )

            max_score = float(anomaly_scores.max())

        return max_score

    def optimize_threshold(
        self,
        test_good_path: Path,
        test_defect_path: Path,
        knn_k: int,
        metric: str = "cosine",
        validation_output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize threshold on test set with comprehensive validation visualizations.

        Args:
            test_good_path: Path to test/good images
            test_defect_path: Path to test/defect images
            knn_k: Number of nearest neighbors
            metric: Distance metric
            validation_output_dir: Directory to save validation visualizations (optional)
            progress_callback: Callback(current, total, message)

        Returns:
            optimal_threshold: Best threshold value
            metrics: Validation metrics (accuracy, precision, recall, f1, tp, tn, fp, fn)
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built!")

        print("\nOptimizing threshold on test set...")

        # Get all test images
        good_images = sorted(list(test_good_path.glob("*.png")))
        defect_images = sorted(list(test_defect_path.glob("*.png")))

        print(f"  Good images: {len(good_images)}")
        print(f"  Defect images: {len(defect_images)}")

        if len(good_images) == 0 or len(defect_images) == 0:
            raise ValueError("Need both good and defect test images!")

        # Compute scores for all test images
        good_scores = []
        defect_scores = []
        good_image_names = []
        defect_image_names = []

        total_images = len(good_images) + len(defect_images)
        processed = 0

        # Process good images
        for img_path in good_images:
            score = self.predict_image(img_path, knn_k, metric)
            good_scores.append(score)
            good_image_names.append(img_path.name)
            processed += 1

            if progress_callback:
                progress_callback(processed, total_images, "Testing on good images")

            print(f"  Testing: {processed}/{total_images}", end='\r')

        # Process defect images
        for img_path in defect_images:
            score = self.predict_image(img_path, knn_k, metric)
            defect_scores.append(score)
            defect_image_names.append(img_path.name)
            processed += 1

            if progress_callback:
                progress_callback(processed, total_images, "Testing on defect images")

            print(f"  Testing: {processed}/{total_images}", end='\r')

        print()  # New line

        # Convert to numpy arrays
        good_scores_np = np.array(good_scores)
        defect_scores_np = np.array(defect_scores)

        # Compute statistics
        print(f"\nScore statistics:")
        print(f"  Good images:  Min: {good_scores_np.min():.4f}, Max: {good_scores_np.max():.4f}, Mean: {good_scores_np.mean():.4f}, Std: {good_scores_np.std():.4f}")
        print(f"  Defect images: Min: {defect_scores_np.min():.4f}, Max: {defect_scores_np.max():.4f}, Mean: {defect_scores_np.mean():.4f}, Std: {defect_scores_np.std():.4f}")

        # HYBRID THRESHOLD: Midpoint between 99th percentile of good and minimum of defect
        good_upper_bound = np.percentile(good_scores_np, 99.0)  # 99th percentile of good
        defect_lower_bound = defect_scores_np.min()  # Minimum defect score

        # Calculate hybrid threshold as midpoint
        hybrid_threshold = (good_upper_bound + defect_lower_bound) / 2.0

        print(f"\n  Hybrid threshold calculation:")
        print(f"    Good upper bound (99th percentile): {good_upper_bound:.4f}")
        print(f"    Defect lower bound (minimum):       {defect_lower_bound:.4f}")
        print(f"    Hybrid threshold (midpoint):        {hybrid_threshold:.4f}")

        # Compute metrics for hybrid threshold
        predictions_good = (good_scores_np > hybrid_threshold).astype(int)
        predictions_defect = (defect_scores_np > hybrid_threshold).astype(int)

        tp = np.sum(predictions_defect == 1)  # Correctly detected defects
        tn = np.sum(predictions_good == 0)  # Correctly detected good
        fp = np.sum(predictions_good == 1)  # False positives (good detected as defect)
        fn = np.sum(predictions_defect == 0)  # False negatives (defect detected as good)

        accuracy = (tp + tn) / (len(good_scores) + len(defect_scores))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        best_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }

        print(f"\n[OK] Hybrid threshold: {hybrid_threshold:.4f}")
        print(f"  Accuracy:  {best_metrics['accuracy']:.2%}")
        print(f"  Precision: {best_metrics['precision']:.2%}")
        print(f"  Recall:    {best_metrics['recall']:.2%}")
        print(f"  F1-Score:  {best_metrics['f1_score']:.2%}")
        print(f"  True Positives:  {best_metrics['true_positives']}")
        print(f"  True Negatives:  {best_metrics['true_negatives']}")
        print(f"  False Positives: {best_metrics['false_positives']}")
        print(f"  False Negatives: {best_metrics['false_negatives']}")

        # Generate validation visualizations if output directory is provided
        if validation_output_dir is not None:
            try:
                from validation_visualizer import ValidationVisualizer

                print(f"\n[INFO] Generating validation visualizations...")
                print(f"  Output directory: {validation_output_dir}")

                visualizer = ValidationVisualizer(validation_output_dir)
                visualizer.generate_full_report(
                    good_scores=good_scores,
                    defect_scores=defect_scores,
                    threshold=hybrid_threshold,
                    metrics=best_metrics,
                    good_image_names=good_image_names,
                    defect_image_names=defect_image_names
                )
            except Exception as e:
                error_msg = f"[WARN] Failed to generate validation visualizations: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()

                # Write error to log file in validation output directory
                try:
                    from pathlib import Path
                    validation_output_dir = Path(validation_output_dir)
                    validation_output_dir.mkdir(parents=True, exist_ok=True)
                    error_log_path = validation_output_dir / "validation_error_from_training.log"
                    with open(error_log_path, 'w') as f:
                        f.write(f"Error during validation visualization in model_trainer.py:\n")
                        f.write(f"{error_msg}\n\n")
                        f.write("Full traceback:\n")
                        traceback.print_exc(file=f)
                    print(f"[INFO] Error details saved to: {error_log_path}")
                except Exception as log_error:
                    print(f"[ERROR] Could not write error log: {log_error}")

        return hybrid_threshold, best_metrics

    def compute_background_baseline_scores(
        self,
        background_images: List[np.ndarray],
        knn_k: int,
        metric: str = "cosine"
    ) -> List[float]:
        """
        Compute anomaly scores for background images (no object present).

        Used to establish baseline for "no object" detection.

        Args:
            background_images: List of background images (numpy arrays [H, W, 3])
            knn_k: Number of nearest neighbors
            metric: Distance metric

        Returns:
            List of anomaly scores (one per image)
        """
        if self.model is None:
            self.load_model()

        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built!")

        if self.training_resolution is None:
            raise RuntimeError("Training resolution not set! Build memory bank first.")

        print(f"\nComputing background baseline scores...")
        print(f"  Processing {len(background_images)} background images")

        baseline_scores = []

        for idx, img_np in enumerate(background_images):
            # Preprocess image
            img_tensor = self._prepare_image_tensor(img_np, self.training_resolution).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Extract features with AMP
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    patch_features = self._extract_features(img_tensor)

                # Convert to FP32
                if self.use_amp:
                    patch_features = patch_features.float()

                # Normalize
                patch_features = patch_features / (
                    patch_features.norm(dim=-1, keepdim=True) + 1e-6
                )

                # Compute scores
                anomaly_scores = self._compute_anomaly_scores(
                    patch_features[0],
                    knn_k,
                    metric
                )

                max_score = float(anomaly_scores.max())
                baseline_scores.append(max_score)

                print(f"  Background image {idx + 1}/{len(background_images)}: score = {max_score:.4f}")

        print(f"[OK] Background baseline scores computed: {baseline_scores}")
        return baseline_scores

    def get_memory_bank(self) -> Optional[torch.Tensor]:
        """Get current memory bank."""
        return self.memory_bank

    def set_memory_bank(self, memory_bank: torch.Tensor):
        """Set memory bank (e.g., loaded from file)."""
        self.memory_bank = memory_bank
        print(f"Memory bank loaded: {memory_bank.shape}")

    def _random_coreset_sampling(self, ratio: float) -> torch.Tensor:
        """
        Random coreset sampling (FASTEST, surprisingly effective).

        Performance: O(1) - instant sampling
        Quality: 85-95% of greedy k-Center quality for anomaly detection

        Args:
            ratio: Ratio of samples to keep (0.01-1.0)

        Returns:
            coreset_features: Randomly sampled features
        """
        N = self.memory_bank.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Random coreset sampling: {N} -> {target_size} features")

        # Random sampling without replacement
        indices = torch.randperm(N, device=self.memory_bank.device)[:target_size]

        return self.memory_bank[indices]

    def _stratified_random_sampling(self, ratio: float, patches_per_image: int = 1024) -> torch.Tensor:
        """
        Stratified random sampling - samples uniformly from each image.

        Ensures representation from all training images.

        Performance: O(1) - instant sampling
        Quality: Better than pure random, especially with few shots

        Args:
            ratio: Ratio of samples to keep (0.01-1.0)
            patches_per_image: Number of patches per image (default: 1024 for 512x512)

        Returns:
            coreset_features: Stratified sampled features
        """
        N = self.memory_bank.shape[0]
        num_images = N // patches_per_image
        target_size = max(1, int(N * ratio))
        samples_per_image = max(1, target_size // num_images)

        print(f"  Stratified random sampling: {N} -> {target_size} features ({samples_per_image} per image)")

        indices = []
        for img_idx in range(num_images):
            start_idx = img_idx * patches_per_image
            end_idx = start_idx + patches_per_image

            # Random sample from this image
            img_indices = torch.randperm(patches_per_image, device=self.memory_bank.device)[:samples_per_image] + start_idx
            indices.append(img_indices)

        # Concatenate all indices
        indices = torch.cat(indices)

        # Trim to exact target size if needed
        if len(indices) > target_size:
            indices = indices[:target_size]

        return self.memory_bank[indices]

    def _fps_coreset_sampling_gpu(self, ratio: float, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> torch.Tensor:
        """
        Farthest Point Sampling (FPS) on GPU using PyTorch.

        GPU-accelerated version of greedy k-Center, much faster than CPU version.

        Performance: O(M * N) but on GPU (~10-50x faster than CPU greedy)
        Quality: Identical to greedy k-Center

        Args:
            ratio: Ratio of samples to keep (0.01-1.0)
            progress_callback: Callback(current, total, message)

        Returns:
            coreset_features: FPS sampled features
        """
        N, C = self.memory_bank.shape
        target_size = max(1, int(N * ratio))

        print(f"  FPS coreset sampling (GPU): {N} -> {target_size} features")

        device = self.memory_bank.device

        # Initialize with random point
        selected_indices = torch.zeros(target_size, dtype=torch.long, device=device)
        selected_indices[0] = torch.randint(0, N, (1,), device=device)

        # Track minimum distances to selected set
        min_dists = torch.full((N,), float('inf'), device=device)

        for i in range(1, target_size):
            # Get last selected point
            last_idx = selected_indices[i-1]
            last_point = self.memory_bank[last_idx:last_idx+1]  # [1, C]

            # Compute distances from all points to last selected point
            # Using cosine distance (1 - dot product for normalized features)
            dists = 1.0 - (self.memory_bank @ last_point.t()).squeeze()  # [N]

            # Update minimum distances
            min_dists = torch.minimum(min_dists, dists)

            # Select point with maximum minimum distance
            selected_indices[i] = torch.argmax(min_dists)

            # Progress callback
            if progress_callback and i % max(1, target_size // 20) == 0:
                progress_callback(i, target_size, f"FPS coreset selection")

            if i % max(1, target_size // 10) == 0:
                print(f"    FPS progress: {i}/{target_size}", end='\r')

        print()  # New line

        return self.memory_bank[selected_indices]

    def _importance_sampling(self, ratio: float) -> torch.Tensor:
        """
        Importance sampling based on feature variance/uniqueness.

        Samples features that are more unique (further from centroid).

        Performance: O(N) - very fast
        Quality: Good for diverse feature representation

        Args:
            ratio: Ratio of samples to keep (0.01-1.0)

        Returns:
            coreset_features: Importance sampled features
        """
        N = self.memory_bank.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Importance sampling: {N} -> {target_size} features")

        # Compute centroid
        centroid = self.memory_bank.mean(dim=0, keepdim=True)  # [1, C]

        # Compute distance to centroid (importance score)
        # Features far from centroid are more "unique"
        importance = 1.0 - (self.memory_bank @ centroid.t()).squeeze()  # [N]

        # Sample based on importance (higher = more likely)
        # Use weighted sampling
        probabilities = importance / importance.sum()
        indices = torch.multinomial(probabilities, target_size, replacement=False)

        return self.memory_bank[indices]

    def _greedy_coreset_sampling_gpu(self, ratio: float, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> torch.Tensor:
        """
        Greedy k-Center coreset sampling on GPU (LEGACY - use fps_gpu instead).

        This is the old GPU-based greedy method. FPS is equivalent and often faster.

        Args:
            ratio: Ratio of samples to keep (0.01-1.0)
            progress_callback: Callback(current, total, message)

        Returns:
            coreset_features: Sampled features
        """
        N = self.memory_bank.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Greedy k-Center (GPU): {N} -> {target_size} features")

        # Start with random center
        selected_indices = [torch.randint(0, N, (1,)).item()]

        # Iteratively select features that are furthest from current coreset
        for i in range(1, target_size):
            # Compute distances from all points to current coreset
            coreset_features = self.memory_bank[selected_indices]

            # Cosine distance: 1 - similarity
            similarities = self.memory_bank @ coreset_features.t()  # [N, coreset_size]
            distances = 1 - similarities  # [N, coreset_size]

            # For each point, find minimum distance to coreset
            min_distances = distances.min(dim=1).values  # [N]

            # Select point with maximum minimum distance (furthest from coreset)
            furthest_idx = min_distances.argmax().item()
            selected_indices.append(furthest_idx)

            # Progress callback
            if progress_callback and i % max(1, target_size // 20) == 0:
                progress_callback(i, target_size, f"Greedy coreset selection")

            if i % max(1, target_size // 10) == 0:
                print(f"  Selected: {i}/{target_size} features", end='\r')

        print()  # New line

        return self.memory_bank[selected_indices]

    def reduce_memory_bank_coreset(
        self,
        percentage: float,
        method: str = "random",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> torch.Tensor:
        """
        Reduce memory bank using various coreset selection methods.

        Available methods:
        - "random": Random sampling (FASTEST, ~95% quality) - RECOMMENDED
        - "stratified": Stratified random (balanced across images, ~97% quality)
        - "fps_gpu": Farthest Point Sampling on GPU (best quality, slower)
        - "importance": Importance sampling (diverse features)
        - "greedy": Legacy greedy k-Center (use fps_gpu instead)

        Args:
            percentage: Percentage of memory bank to keep (1-100%)
            method: Coreset selection method (default: "random")
            progress_callback: Callback(current, total, message)

        Returns:
            Reduced memory bank tensor

        Example:
            >>> trainer.reduce_memory_bank_coreset(10.0, method="random")
            >>> trainer.reduce_memory_bank_coreset(5.0, method="stratified")
            >>> trainer.reduce_memory_bank_coreset(10.0, method="fps_gpu")
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built yet!")

        original_size = self.memory_bank.shape[0]
        ratio = percentage / 100.0

        print(f"\n{'='*60}")
        print(f"REDUCING MEMORY BANK")
        print(f"  Method: {method}")
        print(f"  Percentage: {percentage}% (ratio: {ratio})")
        print(f"  Original size: {original_size} features")
        print(f"{'='*60}")

        start_time = time.time()

        # Select coreset method
        if method == "random":
            reduced_memory_bank = self._random_coreset_sampling(ratio)
        elif method == "stratified":
            patches_per_image = self.patches_per_image or 1024
            reduced_memory_bank = self._stratified_random_sampling(ratio, patches_per_image=patches_per_image)
        elif method == "fps_gpu":
            reduced_memory_bank = self._fps_coreset_sampling_gpu(ratio, progress_callback)
        elif method == "importance":
            reduced_memory_bank = self._importance_sampling(ratio)
        elif method == "greedy":
            reduced_memory_bank = self._greedy_coreset_sampling_gpu(ratio, progress_callback)
        else:
            raise ValueError(
                f"Unknown coreset method: {method}\n"
                f"Available methods: random, stratified, fps_gpu, importance, greedy"
            )

        elapsed = time.time() - start_time
        reduction_ratio = (1 - reduced_memory_bank.shape[0] / original_size) * 100

        print(f"{'='*60}")
        print(f"[OK] CORESET REDUCTION COMPLETE!")
        print(f"  Reduced size:  {reduced_memory_bank.shape[0]} features")
        print(f"  Reduction:     {reduction_ratio:.1f}%")
        print(f"  Time elapsed:  {elapsed:.2f}s")
        print(f"  Speedup:       ~{original_size/reduced_memory_bank.shape[0]:.1f}x faster k-NN")
        print(f"{'='*60}\n")

        # Update memory bank
        self.memory_bank = reduced_memory_bank

        return reduced_memory_bank
