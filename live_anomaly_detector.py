"""
Live Anomaly Detector using DINOv3 + Zero-Shot k-NN
==================================================

Real-time anomaly detection with DINOv3 multi-layer features.

Features:
- DINOv3 multi-layer feature extraction (configurable layers)
- DINOv3 register token handling (skip 1 CLS + 4 register tokens)
- Torch k-NN search (cosine distance)
- Optional Coreset sampling for memory efficiency
- Bilinear heatmap upsampling

Based on AD-DINOv3_ZERO-SHOT approach.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import time
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoModel
from PIL import Image

from motion_filter import MotionHysteresisFilter

# FAISS will be lazy-loaded when needed to avoid conflicts with IDS Peak
# Do NOT import faiss at module level!
FAISS_AVAILABLE = None  # Will be checked on first use
SCANN_AVAILABLE = None  # Will be checked on first use


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


def _ensure_faiss():
    """
    Lazy-load FAISS when needed to avoid conflicts with IDS Peak.

    FAISS uses SWIG bindings which can conflict with IDS Peak's native C++ bindings.
    We only import FAISS when actually needed (during training/index building).

    Returns:
        faiss module if available

    Raises:
        RuntimeError if FAISS is required but not available
    """
    global FAISS_AVAILABLE

    if FAISS_AVAILABLE is None:
        try:
            import faiss
            FAISS_AVAILABLE = faiss
            print("  [OK] FAISS loaded successfully (lazy-loaded)")
        except ImportError as e:
            FAISS_AVAILABLE = False
            print(f"  [WARN] FAISS not available: {e}")
            print("  Will use torch.cdist for k-NN search (slower)")

    if FAISS_AVAILABLE is False:
        raise RuntimeError(
            "FAISS is required for this operation but not available.\n"
            "Install with: pip install faiss-gpu (CUDA) or pip install faiss-cpu"
        )

    return FAISS_AVAILABLE


def _ensure_scann():
    """
    Lazy-load ScaNN when needed to avoid hard dependency.

    Returns:
        scann module if available

    Raises:
        RuntimeError if ScaNN is required but not available
    """
    global SCANN_AVAILABLE

    if SCANN_AVAILABLE is None:
        try:
            import scann
            SCANN_AVAILABLE = scann
            print("  [OK] ScaNN loaded successfully (lazy-loaded)")
        except ImportError as e:
            SCANN_AVAILABLE = False
            print(f"  [WARN] ScaNN not available: {e}")

    if SCANN_AVAILABLE is False:
        raise RuntimeError(
            "ScaNN is required for this operation but not available.\n"
            "Install with: pip install scann"
        )

    return SCANN_AVAILABLE


class LiveAnomalyDetector:
    """
    Real-time anomaly detector with DINOv3 + Zero-Shot k-NN.

    Features:
    - DINOv3 single-layer feature extraction (last layer only)
    - Torch k-NN search (cosine distance)
    - Optional Coreset sampling for memory efficiency
    - Bilinear heatmap upsampling with align_corners=True
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        reference_path: Optional[Path] = None,
        shots: int = 20,
        knn_k: int = 5,
        metric: str = "cosine",  # Changed default to 'cosine' to match working code
        use_faiss: bool = True,  # Deprecated: use ann_backend instead
        ann_backend: Optional[str] = None,  # "faiss_gpu", "faiss_cpu", "scann", "torch"
        use_coreset: bool = False,
        coreset_ratio: float = 0.1,
        coreset_method: str = "random",  # Coreset method: "random", "stratified", "fps_gpu", "importance", "greedy"
        use_adaptive_knn: bool = False,  # Disabled for real-time performance (50-100ms overhead)
        adaptive_k_min: int = 3,
        adaptive_k_max: int = 9,
        use_spatial_smoothing: bool = False,  # Disabled for real-time performance (80-150ms overhead)
        spatial_filter_d: int = 5,  # Bilateral filter diameter
        spatial_filter_sigma_color: float = 75.0,  # Bilateral filter sigma color
        spatial_filter_sigma_space: float = 75.0,  # Bilateral filter sigma space
        selected_layers: Optional[List[int]] = None,  # Which layers to extract (default: [-1] = last only)
        training_resolution: int = 512,  # Training resolution (must match memory bank!)
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
        use_pinned_memory: bool = False,
        # Motion Detection Parameters (FPS-adaptive)
        enable_motion_filter: bool = False,
        motion_high_threshold: float = 0.05,
        motion_low_threshold: float = 0.01,
        motion_stabilization_time: float = 2.0,
        motion_learning_time: float = 10.0,
    ):
        """
        Initialize the anomaly detector (MULTI-LAYER configurable).

        Args:
            model_name: DINOv3 model (ViT-S/B/L)
            reference_path: Path to reference images
            shots: Number of reference samples
            knn_k: Number of nearest neighbors (used if adaptive_knn=False)
            metric: Distance metric ('l2' or 'cosine')
            use_faiss: Use FAISS GPU for fast k-NN (NOT recommended with IDS Peak!)
            ann_backend: ANN backend ("faiss_gpu", "faiss_cpu", "scann", "torch")
            use_coreset: Use coreset sampling to reduce memory bank size
            coreset_ratio: Ratio of samples to keep (0.1 = 10%)
            use_adaptive_knn: Use adaptive k-NN based on local density
            adaptive_k_min: Minimum k for dense regions (adaptive k-NN)
            adaptive_k_max: Maximum k for sparse regions (adaptive k-NN)
            use_spatial_smoothing: Apply spatial-aware smoothing (bilateral filter)
            spatial_filter_d: Bilateral filter diameter
            spatial_filter_sigma_color: Bilateral filter sigma in color space
            spatial_filter_sigma_space: Bilateral filter sigma in coordinate space
            selected_layers: Which layers to extract features from (default: [-1] = last layer only)
            training_resolution: Training resolution in pixels (must match memory bank training!)
            device: Torch device (auto-detect if None)
            cache_dir: Model cache directory
        """
        self.model_name = model_name
        self.shots = shots
        self.knn_k = knn_k
        self.metric = metric
        if ann_backend is None:
            ann_backend = "faiss_gpu" if use_faiss else "torch"
        ann_backend = ann_backend.lower()
        self.ann_backend = ann_backend
        self.use_faiss = ann_backend.startswith("faiss")
        self.use_coreset = use_coreset
        self.coreset_ratio = coreset_ratio
        self.coreset_method = coreset_method
        self.use_adaptive_knn = use_adaptive_knn
        self.adaptive_k_min = adaptive_k_min
        self.adaptive_k_max = adaptive_k_max
        self.use_spatial_smoothing = use_spatial_smoothing
        self.spatial_filter_d = spatial_filter_d
        self.spatial_filter_sigma_color = spatial_filter_sigma_color
        self.spatial_filter_sigma_space = spatial_filter_sigma_space
        self.selected_layers = selected_layers if selected_layers is not None else [-1]
        self.use_pinned_memory = use_pinned_memory

        # Motion Detection Parameters
        self.enable_motion_filter = enable_motion_filter
        self.motion_high_threshold = motion_high_threshold
        self.motion_low_threshold = motion_low_threshold
        self.motion_stabilization_time = motion_stabilization_time
        self.motion_learning_time = motion_learning_time

        # Device setup
        self.device = device if device else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        print("=" * 60)
        print("DINOv3 Anomaly Detector (Single Resolution)")
        print("=" * 60)
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Training Resolution: {training_resolution}x{training_resolution}")
        print(f"  Selected Layers: {self.selected_layers}")
        if use_adaptive_knn:
            print(f"  k-NN: Adaptive (k={adaptive_k_min}-{adaptive_k_max}), Metric: {metric}")
        else:
            print(f"  k-NN: Fixed (k={knn_k}), Metric: {metric}")
        print(f"  ANN Backend: {self.ann_backend}")
        print(f"  Coreset: {'Enabled (ratio=' + str(coreset_ratio) + ')' if use_coreset else 'Disabled'}")
        print(f"  Spatial Smoothing: {'Enabled (Bilateral Filter)' if use_spatial_smoothing else 'Disabled'}")
        print(f"  Motion Filter: {'Enabled (hysteresis)' if enable_motion_filter else 'Disabled'}")

        # Model cache
        if cache_dir is None:
            cache_dir = Path("models")
        self.cache_dir = cache_dir

        # State
        self.model = None  # DINOv3
        self.memory_bank = None  # k-NN memory bank (single tensor [N, C])
        self.faiss_index = None  # FAISS index for fast k-NN (if use_faiss=True)
        self.scann_index = None  # ScaNN index for ANN search
        self.memory_bank_normalized = False
        self.is_ready = False

        # Performance tracking
        self.inference_times = []

        # FPS tracking for motion filter
        self.frame_times = deque(maxlen=30)  # Last 30 frame timestamps
        self.current_fps = 10.0  # Initial FPS estimate

        # Motion Detection Filter
        self.motion_filter = None
        if self.enable_motion_filter:
            self.motion_filter = MotionHysteresisFilter(
                motion_high_threshold=motion_high_threshold,
                motion_low_threshold=motion_low_threshold,
                stabilization_time_sec=motion_stabilization_time,
                learning_time_sec=motion_learning_time,
                estimated_fps=self.current_fps
            )
            print(f"  [OK] Motion filter initialized with FPS={self.current_fps:.1f}")

        # Resolution settings (must match training!)
        self.training_resolution = training_resolution
        self.heatmap_size = training_resolution  # Heatmap matches training resolution
        self.preprocess_size = (training_resolution, training_resolution)
        self._preprocess_buffer = None
        self._preprocess_mean = None
        self._preprocess_std = None

        # Load model
        self._load_model()

        # Build memory bank if reference path provided
        if reference_path:
            self._build_memory_bank(reference_path)

        print("=" * 60)
        print("[OK] DINOv3 Detector Ready!")
        print("=" * 60)

    def _get_preprocess_buffers(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate/reuse preprocessing buffers on the current device."""
        if (
            self._preprocess_buffer is None
            or self._preprocess_buffer.device != self.device
        ):
            self._preprocess_buffer = torch.empty(
                (1, 3, self.preprocess_size[1], self.preprocess_size[0]),
                device=self.device,
                dtype=torch.float32
            )
            self._preprocess_mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
                device=self.device
            ).view(1, 3, 1, 1)
            self._preprocess_std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
                device=self.device
            ).view(1, 3, 1, 1)
        return self._preprocess_buffer, self._preprocess_mean, self._preprocess_std

    def _preprocess_image_cpu(self, image: np.ndarray) -> torch.Tensor:
        """CPU preprocessing baseline (resize + normalize)."""
        if image.shape[:2] != self.preprocess_size:
            resized = cv2.resize(image, self.preprocess_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image

        img_float = resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def benchmark_preprocess(self, image: np.ndarray, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark CPU vs GPU preprocessing (ms per frame).

        Returns:
            Dict with average ms per frame for CPU and GPU paths.
        """
        cpu_times = []
        gpu_times = []

        for _ in range(iterations):
            t_start = time.perf_counter()
            _ = self._preprocess_image_cpu(image)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cpu_times.append((time.perf_counter() - t_start) * 1000.0)

        for _ in range(iterations):
            t_start = time.perf_counter()
            _ = self.preprocess_image(image)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            gpu_times.append((time.perf_counter() - t_start) * 1000.0)

        return {
            "cpu_preprocess_ms": float(np.mean(cpu_times)),
            "gpu_preprocess_ms": float(np.mean(gpu_times))
        }

    def _load_model(self):
        """Load DINOv3 model from local directory only (no HuggingFace cache)."""
        print(f"Loading DINOv3 model: {self.model_name}...")

        # Use simple directory format: models/facebook_dinov3-vitb16-pretrain-lvd1689m
        model_cache_dir = self.cache_dir / self.model_name.replace("/", "_")

        # Try to load from local cache first (preferred)
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
                    self.model = AutoModel.from_pretrained(self.model_name)
                    model_cache_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(model_cache_dir)
                    print(f"  [OK] Model downloaded and saved to: {model_cache_dir}")
                except Exception as download_error:
                    raise RuntimeError(
                        f"Failed to load model '{self.model_name}'. "
                        f"Local cache failed: {e}. "
                        f"Download failed: {download_error}. "
                        f"\n\nPlease check:\n"
                        f"1. Model exists in cache: {model_cache_dir}\n"
                        f"2. Model name is correct: {self.model_name}\n"
                        f"3. Internet connection for download"
                    )
        else:
            print(f"  -> Model not found locally, downloading from Hugging Face...")
            try:
                self.model = AutoModel.from_pretrained(self.model_name)
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_cache_dir)
                print(f"  [OK] Model downloaded and saved to: {model_cache_dir}")
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to download model '{self.model_name}': {download_error}\n\n"
                    f"Please check:\n"
                    f"1. Model name is correct (e.g., 'facebook/dinov3-vitl16-pretrain-lvd1689m')\n"
                    f"2. Internet connection\n"
                    f"3. Hugging Face availability"
                )

        self.model.to(self.device)
        self.model.eval()

        print("[OK] DINOv3 Model loaded and ready")

    def _extract_features(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract DINOv3 patch features from multiple layers (MULTI-LAYER).

        DINOv3 token structure: [CLS, REG1, REG2, REG3, REG4, PATCH_TOKENS...]
        We skip the first 5 tokens (1 CLS + 4 registers) to get spatial patches.

        Args:
            images: Input images [B, 3, 512, 512]

        Returns:
            patch_features: Patch tokens [B, N_patches, C_total]
                           where C_total = sum of feature dims from selected layers
        """
        with torch.no_grad():
            outputs = self.model(images, output_hidden_states=True)

            # Extract features from selected layers
            layer_features = []

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
                expected_patches = (self.training_resolution // 16) ** 2  # e.g., 32x32 = 1024 for 512x512

                # Skip CLS + register tokens to extract spatial patches
                if N_total == expected_patches + 5:
                    # Standard DINOv3: 1 CLS + 4 registers + 1024 patches = 1029
                    patch_features = hidden_states[:, 5:, :]
                elif N_total > expected_patches + 5:
                    # More tokens than expected, extract exact patch count
                    patch_features = hidden_states[:, 5:5+expected_patches, :]
                elif N_total == expected_patches + 1:
                    # No registers (older models), only CLS token
                    patch_features = hidden_states[:, 1:, :]
                else:
                    # Fallback: skip CLS only
                    patch_features = hidden_states[:, 1:, :]

                layer_features.append(patch_features)

            # Concatenate features from all selected layers
            if len(layer_features) == 1:
                return layer_features[0]
            else:
                return torch.cat(layer_features, dim=-1)  # Concatenate along feature dimension

    def _random_coreset_sampling(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Random coreset sampling (FASTEST, surprisingly effective).

        Performance: O(1) - instant sampling
        Quality: 85-95% of greedy k-Center quality for anomaly detection

        Args:
            features: Feature vectors [N, C]
            ratio: Ratio of samples to keep (0.1 = 10%)

        Returns:
            coreset_features: Randomly sampled features [N*ratio, C]
        """
        N = features.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Random coreset sampling: {N} -> {target_size} features")
        print(f"  Memory bank device: {features.device}")

        # FIX: Ensure we use GPU if available for fast random sampling
        device = features.device
        if device.type == 'cpu' and torch.cuda.is_available():
            print(f"  [WARN] Memory bank on CPU! Moving to GPU for faster sampling...")
            features = features.cuda()
            device = features.device

        # Random sampling without replacement
        indices = torch.randperm(N, device=device)[:target_size]

        return features[indices]

    def _stratified_random_sampling(self, features: torch.Tensor, ratio: float, patches_per_image: int = 1024) -> torch.Tensor:
        """
        Stratified random sampling - samples uniformly from each image.

        Ensures representation from all training images.

        Performance: O(1) - instant sampling
        Quality: Better than pure random, especially with few shots

        Args:
            features: Feature vectors [N, C] where N = num_images * patches_per_image
            ratio: Ratio of samples to keep (0.1 = 10%)
            patches_per_image: Number of patches per image (default: 1024 for 512x512)

        Returns:
            coreset_features: Stratified sampled features [N*ratio, C]
        """
        N = features.shape[0]
        num_images = N // patches_per_image
        target_size = max(1, int(N * ratio))
        samples_per_image = max(1, target_size // num_images)

        print(f"  Stratified random sampling: {N} -> {target_size} features ({samples_per_image} per image)")

        indices = []
        for img_idx in range(num_images):
            start_idx = img_idx * patches_per_image
            end_idx = start_idx + patches_per_image

            # Random sample from this image
            img_indices = torch.randperm(patches_per_image, device=features.device)[:samples_per_image] + start_idx
            indices.append(img_indices)

        # Concatenate all indices
        indices = torch.cat(indices)

        # Trim to exact target size if needed
        if len(indices) > target_size:
            indices = indices[:target_size]

        return features[indices]

    def _fps_coreset_sampling_gpu(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Farthest Point Sampling (FPS) on GPU using PyTorch.

        GPU-accelerated version of greedy k-Center, much faster than CPU version.

        Performance: O(M * N) but on GPU (~10-50x faster than CPU greedy)
        Quality: Identical to greedy k-Center

        Args:
            features: Feature vectors [N, C]
            ratio: Ratio of samples to keep (0.1 = 10%)

        Returns:
            coreset_features: FPS sampled features [N*ratio, C]
        """
        N, C = features.shape
        target_size = max(1, int(N * ratio))

        print(f"  FPS coreset sampling (GPU): {N} -> {target_size} features")

        device = features.device

        # Initialize with random point
        selected_indices = torch.zeros(target_size, dtype=torch.long, device=device)
        selected_indices[0] = torch.randint(0, N, (1,), device=device)

        # Track minimum distances to selected set
        min_dists = torch.full((N,), float('inf'), device=device)

        for i in range(1, target_size):
            # Get last selected point
            last_idx = selected_indices[i-1]
            last_point = features[last_idx:last_idx+1]  # [1, C]

            # Compute distances from all points to last selected point
            # Using cosine distance (1 - dot product for normalized features)
            dists = 1.0 - (features @ last_point.t()).squeeze()  # [N]

            # Update minimum distances
            min_dists = torch.minimum(min_dists, dists)

            # Select point with maximum minimum distance
            selected_indices[i] = torch.argmax(min_dists)

            # Progress indicator
            if i % max(1, target_size // 10) == 0:
                print(f"    FPS progress: {i}/{target_size}", end='\r')

        print()  # New line

        return features[selected_indices]

    def _importance_sampling(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Importance sampling based on feature variance/uniqueness.

        Samples features that are more unique (further from centroid).

        Performance: O(N) - very fast
        Quality: Good for diverse feature representation

        Args:
            features: Feature vectors [N, C]
            ratio: Ratio of samples to keep (0.1 = 10%)

        Returns:
            coreset_features: Importance sampled features [N*ratio, C]
        """
        N = features.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Importance sampling: {N} -> {target_size} features")

        # Compute centroid
        centroid = features.mean(dim=0, keepdim=True)  # [1, C]

        # Compute distance to centroid (importance score)
        # Features far from centroid are more "unique"
        importance = 1.0 - (features @ centroid.t()).squeeze()  # [N]

        # Sample based on importance (higher = more likely)
        # FIX: Use topk instead of multinomial (multinomial is EXTREMELY slow with replacement=False!)
        _, indices = torch.topk(importance, target_size, largest=True)

        return features[indices]

    def _greedy_coreset_sampling(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Greedy k-Center coreset sampling (LEGACY - slow on CPU).

        NOTE: This is the old CPU-based method. Use _fps_coreset_sampling_gpu() instead.

        Args:
            features: Feature vectors [N, C]
            ratio: Ratio of samples to keep (0.1 = 10%)

        Returns:
            coreset_features: Sampled features [N*ratio, C]
        """
        N = features.shape[0]
        target_size = max(1, int(N * ratio))

        print(f"  Greedy k-Center (CPU): {N} -> {target_size} features [SLOW - consider GPU FPS instead]")

        # Start with random center
        indices = [torch.randint(0, N, (1,)).item()]
        features_np = features.cpu().numpy()

        # Greedy selection
        for _ in range(target_size - 1):
            # Compute distances to closest selected point
            selected = features_np[indices]
            dists = np.linalg.norm(
                features_np[:, None, :] - selected[None, :, :],
                axis=2
            ).min(axis=1)

            # Select point with maximum minimum distance
            new_idx = dists.argmax()
            indices.append(new_idx)

        return features[indices]

    def apply_coreset(
        self,
        method: str = "random",
        ratio: float = 0.1,
        patches_per_image: int = 1024
    ) -> torch.Tensor:
        """
        Apply coreset sampling to reduce memory bank size.

        Available methods:
        - "random": Random sampling (FASTEST, ~95% quality)
        - "stratified": Stratified random (balanced across images)
        - "fps_gpu": Farthest Point Sampling on GPU (slower, best quality)
        - "importance": Importance sampling (unique features)
        - "greedy": Legacy greedy k-Center on CPU (SLOW, not recommended)

        Recommended: "random" for real-time, "fps_gpu" for best accuracy

        Args:
            method: Sampling method
            ratio: Ratio to keep (0.1 = 10%)
            patches_per_image: Patches per image (for stratified)

        Returns:
            Reduced memory bank
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built yet!")

        print(f"\n{'='*60}")
        print(f"APPLYING CORESET REDUCTION")
        print(f"  Method: {method}")
        print(f"  Ratio: {ratio} ({ratio*100:.0f}%)")
        print(f"  Memory bank size BEFORE: {self.memory_bank.shape}")
        print(f"  Memory bank device: {self.memory_bank.device}")
        print(f"  ann_backend: {self.ann_backend}")
        print(f"{'='*60}")

        t_start = time.time()

        if method == "random":
            reduced = self._random_coreset_sampling(self.memory_bank, ratio)
        elif method == "stratified":
            reduced = self._stratified_random_sampling(self.memory_bank, ratio, patches_per_image)
        elif method == "fps_gpu":
            reduced = self._fps_coreset_sampling_gpu(self.memory_bank, ratio)
        elif method == "importance":
            reduced = self._importance_sampling(self.memory_bank, ratio)
        elif method == "greedy":
            reduced = self._greedy_coreset_sampling(self.memory_bank, ratio)
        else:
            raise ValueError(f"Unknown coreset method: {method}")

        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"CORESET REDUCTION COMPLETED!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Original: {self.memory_bank.shape[0]} features")
        print(f"  Reduced:  {reduced.shape[0]} features ({ratio*100:.1f}%)")
        print(f"  Expected k-NN speedup: ~{self.memory_bank.shape[0]/reduced.shape[0]:.1f}x")
        print(f"{'='*60}")

        self.memory_bank = reduced

        # Normalize memory bank once and rebuild ANN index if enabled
        self._normalize_memory_bank()
        self._build_ann_index()

        return reduced

    def _build_faiss_index(self, features: torch.Tensor, use_gpu: bool):
        """
        Build FAISS index for fast k-NN search (single-layer).

        Args:
            features: Feature vectors [N, C]

        Returns:
            FAISS index (CPU or GPU)
        """
        # Lazy-load FAISS when actually needed
        faiss = _ensure_faiss()

        N, C = features.shape
        features_np = features.cpu().numpy().astype('float32')

        if self.metric == 'cosine':
            index = faiss.IndexFlatIP(C)  # Inner product (cosine after normalization)
        else:
            index = faiss.IndexFlatL2(C)  # L2 distance

        # Move to GPU if available
        if use_gpu and self.device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"  [OK] FAISS GPU index created ({N} vectors, {C} dims)")
            except Exception as e:
                print(f"  [WARN] FAISS GPU failed ({e}), using CPU")
                print(f"  [OK] FAISS CPU index created ({N} vectors, {C} dims)")
        else:
            print(f"  [OK] FAISS CPU index created ({N} vectors, {C} dims)")

        index.add(features_np)
        return index

    def _build_scann_index(self, features: torch.Tensor):
        """
        Build ScaNN index for ANN search.

        Args:
            features: Feature vectors [N, C]

        Returns:
            ScaNN searcher
        """
        scann = _ensure_scann()

        features_np = features.cpu().numpy().astype('float32')
        distance = "dot_product" if self.metric == 'cosine' else "squared_l2"

        searcher = scann.scann_ops_pybind.builder(
            features_np,
            self.knn_k,
            distance
        ).brute_force().build()

        print(f"  [OK] ScaNN index created ({features_np.shape[0]} vectors, {features_np.shape[1]} dims)")
        return searcher

    def _normalize_memory_bank(self):
        """Normalize memory bank once for cosine similarity."""
        if self.memory_bank is None:
            return
        if self.metric != "cosine":
            self.memory_bank_normalized = True
            return
        if self.memory_bank_normalized:
            return
        self.memory_bank = self.memory_bank / (self.memory_bank.norm(dim=-1, keepdim=True) + 1e-6)
        self.memory_bank_normalized = True

    def _build_ann_index(self):
        """Build ANN index based on selected backend (once per memory bank)."""
        if self.ann_backend == "torch":
            self.faiss_index = None
            self.scann_index = None
            print(f"\n  [INFO] ANN backend set to torch; using torch.cdist for k-NN (slower)")
            return

        if self.ann_backend in {"faiss_gpu", "faiss_cpu"}:
            use_gpu = self.ann_backend == "faiss_gpu"
            print(f"\nRebuilding FAISS index with {self.memory_bank.shape[0]} features...")
            self.faiss_index = self._build_faiss_index(self.memory_bank, use_gpu=use_gpu)
            self.scann_index = None
            print(f"  [OK] FAISS index ready!")
            return

        if self.ann_backend == "scann":
            print(f"\nRebuilding ScaNN index with {self.memory_bank.shape[0]} features...")
            self.scann_index = self._build_scann_index(self.memory_bank)
            self.faiss_index = None
            print(f"  [OK] ScaNN index ready!")
            return

        raise ValueError(f"Unknown ann_backend: {self.ann_backend}")

    def _build_memory_bank(self, reference_path: Path):
        """
        Build memory bank from reference images (SINGLE-LAYER).

        Args:
            reference_path: Path to folder with reference (good) images
                           Expected structure: reference_path/*.png or reference_path/good/*.png
        """
        print(f"Building memory bank from: {reference_path}")

        # Find images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_paths.extend(list(reference_path.glob(ext)))
            if (reference_path / "good").exists():
                image_paths.extend(list((reference_path / "good").glob(ext)))

        if not image_paths:
            raise ValueError(f"No images found in {reference_path}")

        print(f"  Found {len(image_paths)} reference images")
        print(f"  Using {min(self.shots, len(image_paths))} shots")

        # Extract features from reference images
        feats = []

        for i, img_path in enumerate(image_paths[:self.shots]):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)  # RGB format from PIL

            # Preprocess with same pipeline as inference (INTER_AREA + ImageNet normalization)
            img_tensor = self.preprocess_image(img_np)

            # Extract features (single-layer)
            patch_features = self._extract_features(img_tensor)

            # Add to feature list
            feats.append(patch_features[0])

            if i == 0:
                _, N, C = patch_features.shape
                print(f"  Patch tokens per image: {N} (features: {C})")

        # Concatenate to form memory bank [N_patches * shots, C]
        self.memory_bank = torch.cat(feats, dim=0)

        print(f"[OK] Memory bank created: {self.memory_bank.shape}")

        # Normalize memory bank before coreset sampling
        self._normalize_memory_bank()

        # Apply coreset sampling if enabled
        if self.use_coreset:
            self.apply_coreset(method=self.coreset_method, ratio=self.coreset_ratio, patches_per_image=1024)

        # Validate
        if torch.isnan(self.memory_bank).any() or torch.isinf(self.memory_bank).any():
            raise ValueError("Memory bank contains NaN/Inf values")

        # Build ANN index if enabled (done automatically in apply_coreset if coreset was used)
        if self.ann_backend != "torch" and self.faiss_index is None and self.scann_index is None:
            self._build_ann_index()

        self.is_ready = True

    def prepare_memory_bank(self):
        """Normalize memory bank and build ANN index once (for externally provided memory banks)."""
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not set!")

        self._normalize_memory_bank()
        if self.ann_backend in {"faiss_gpu", "faiss_cpu"} and self.faiss_index is not None:
            self.is_ready = True
            return
        if self.ann_backend == "scann" and self.scann_index is not None:
            self.is_ready = True
            return
        self._build_ann_index()
        self.is_ready = True

    def _compute_anomaly_scores_faiss(
        self,
        patch_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly scores using FAISS k-NN search (single-layer).

        Args:
            patch_features: Normalized features [N_patches, C]

        Returns:
            scores: Anomaly score per patch [N_patches]
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built! Call _build_memory_bank first.")

        # FAISS should already be loaded at this point (from _build_memory_bank)
        # But we ensure it's available just in case
        faiss = _ensure_faiss()

        features_np = patch_features.cpu().numpy().astype('float32')

        # k-NN search
        distances, indices = self.faiss_index.search(features_np, self.knn_k)

        # Average k-NN distances
        if self.metric == 'cosine':
            # Convert inner product back to cosine distance
            scores = 1.0 - distances.mean(axis=1)
        else:
            scores = distances.mean(axis=1)

        return torch.from_numpy(scores).to(self.device)

    def _compute_anomaly_scores_scann(
        self,
        patch_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly scores using ScaNN k-NN search.

        Args:
            patch_features: Normalized features [N_patches, C]

        Returns:
            scores: Anomaly score per patch [N_patches]
        """
        if self.scann_index is None:
            raise RuntimeError("ScaNN index not built! Call _build_memory_bank first.")

        features_np = patch_features.cpu().numpy().astype('float32')
        neighbors, distances = self.scann_index.search_batched(features_np)

        if self.metric == 'cosine':
            scores = 1.0 - distances.mean(axis=1)
        else:
            scores = distances.mean(axis=1)

        return torch.from_numpy(scores).to(self.device)

    def _compute_anomaly_scores_torch(
        self,
        patch_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly scores using torch.cdist k-NN (SINGLE-LAYER).

        Args:
            patch_features: Normalized features [N_patches, C]

        Returns:
            scores: Anomaly score per patch [N_patches]
        """
        if self.metric == 'cosine':
            dist = 1.0 - (patch_features @ self.memory_bank.t())
        else:
            dist = torch.cdist(patch_features, self.memory_bank)

        # k-NN averaging
        if self.knn_k > 1:
            topk = torch.topk(dist, k=self.knn_k, largest=False, dim=1).values
            scores = topk.mean(dim=1)
        else:
            scores = dist.min(dim=1).values

        return scores

    def _estimate_local_density(
        self,
        patch_features: torch.Tensor,
        density_k: int = 10
    ) -> torch.Tensor:
        """
        Estimate local feature density for each patch.

        Patches in dense regions (many similar neighbors) get lower density scores,
        patches in sparse regions get higher density scores.

        Args:
            patch_features: Normalized features [N_patches, C]
            density_k: Number of neighbors to estimate density (default: 10)

        Returns:
            density_scores: Density estimate per patch [N_patches]
                           Higher = sparser region, Lower = denser region
        """
        if self.metric == 'cosine':
            # Cosine distance to memory bank
            dist = 1.0 - (patch_features @ self.memory_bank.t())
        else:
            dist = torch.cdist(patch_features, self.memory_bank)

        # Get k-nearest neighbors
        topk_dists = torch.topk(dist, k=min(density_k, dist.shape[1]), largest=False, dim=1).values

        # Average distance to k-NN = density estimate
        # Larger average distance = sparser region
        density_scores = topk_dists.mean(dim=1)

        return density_scores

    def _compute_anomaly_scores_adaptive_knn(
        self,
        patch_features: torch.Tensor,
        k_min: int = 3,
        k_max: int = 9
    ) -> torch.Tensor:
        """
        Compute anomaly scores using adaptive k-NN based on local feature density.

        Dense regions use smaller k (more precise), sparse regions use larger k (more robust).

        Args:
            patch_features: Normalized features [N_patches, C]
            k_min: Minimum k for dense regions (default: 3)
            k_max: Maximum k for sparse regions (default: 9)

        Returns:
            scores: Anomaly score per patch [N_patches]
        """
        # Estimate local density
        density = self._estimate_local_density(patch_features)

        # Normalize density to [0, 1]
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-6)

        # Map density to k values: high density -> small k, low density -> large k
        # Invert: dense (low values) -> k_min, sparse (high values) -> k_max
        k_values = (k_min + (k_max - k_min) * density_norm).long()

        # Compute distances to all memory bank features
        if self.metric == 'cosine':
            dist = 1.0 - (patch_features @ self.memory_bank.t())
        else:
            dist = torch.cdist(patch_features, self.memory_bank)

        # For each patch, compute score with adaptive k
        scores = torch.zeros(patch_features.shape[0], device=self.device)

        for i in range(patch_features.shape[0]):
            k_i = max(1, min(int(k_values[i]), dist.shape[1]))  # Clamp k
            topk = torch.topk(dist[i], k=k_i, largest=False).values
            scores[i] = topk.mean()

        return scores

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess camera frame for inference with ImageNet normalization.

        Args:
            image: RGB image as numpy array [H, W, 3]

        Returns:
            tensor: Preprocessed tensor [1, 3, 512, 512]
        """
        if self.device.type != "cuda":
            return self._preprocess_image_cpu(image)

        buffer, mean, std = self._get_preprocess_buffers()
        img_tensor = torch.from_numpy(image)
        if self.use_pinned_memory:
            img_tensor = img_tensor.pin_memory()

        img_tensor = img_tensor.to(self.device, non_blocking=self.use_pinned_memory)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        if img_tensor.shape[-2:] != self.preprocess_size:
            img_tensor = F.interpolate(
                img_tensor,
                size=self.preprocess_size,
                mode="bilinear",
                align_corners=False
            )

        img_tensor = (img_tensor - mean) / std
        buffer.copy_(img_tensor)
        return buffer

    def predict(
        self,
        image: np.ndarray,
        return_timing: bool = False
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict[str, float]]]:
        """
        Run single-layer k-NN anomaly detection.

        Args:
            image: RGB image [H, W, 3]
            return_timing: Return inference time

        Returns:
            anomaly_map: Anomaly heatmap [heatmap_size, heatmap_size]
            max_score: Maximum anomaly score
            (timing): Dict with timing metrics (if return_timing=True)
        """
        if not self.is_ready:
            raise RuntimeError("Detector not ready!")

        with torch.inference_mode():
            t_start = time.perf_counter()

            # Preprocess
            t_preprocess = time.perf_counter()
            img_tensor = self.preprocess_image(image)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - t_preprocess) * 1000.0

            # Extract features (single-layer)
            patch_features = self._extract_features(img_tensor)

            B, N, C = patch_features.shape
            S = int(N ** 0.5)  # Spatial size (32 for 512x512)

            # Normalize query features once per frame
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)

            # k-NN distance (Adaptive k-NN, FAISS, or torch)
            t_knn_start = time.perf_counter()
            if self.use_adaptive_knn:
                # Adaptive k-NN (always uses torch, not FAISS)
                anomaly_scores = self._compute_anomaly_scores_adaptive_knn(
                    patch_features[0],
                    k_min=self.adaptive_k_min,
                    k_max=self.adaptive_k_max
                )
            elif self.ann_backend in {"faiss_gpu", "faiss_cpu"}:
                if self.faiss_index is None:
                    raise RuntimeError("FAISS index missing. Build memory bank or call apply_coreset first.")
                anomaly_scores = self._compute_anomaly_scores_faiss(patch_features[0])
            elif self.ann_backend == "scann":
                if self.scann_index is None:
                    raise RuntimeError("ScaNN index missing. Build memory bank or call apply_coreset first.")
                anomaly_scores = self._compute_anomaly_scores_scann(patch_features[0])
            else:
                anomaly_scores = self._compute_anomaly_scores_torch(patch_features[0])
            knn_time_ms = (time.perf_counter() - t_knn_start) * 1000.0

            # Reshape to spatial map
            score_map = anomaly_scores.view(S, S).unsqueeze(0).unsqueeze(0)

            # Bilinear upsampling with align_corners=True (like reference code)
            score_map_up = F.interpolate(
                score_map,
                size=self.heatmap_size,
                mode='bilinear',
                align_corners=True
            )[0, 0]

            # Apply spatial-aware smoothing (bilateral filter)
            if self.use_spatial_smoothing:
                # Convert to numpy for cv2.bilateralFilter
                score_map_np = score_map_up.cpu().numpy().astype(np.float32)

                # Normalize to [0, 1] for better bilateral filter behavior
                score_min, score_max = score_map_np.min(), score_map_np.max()
                if score_max > score_min:
                    score_map_normalized = (score_map_np - score_min) / (score_max - score_min)
                else:
                    score_map_normalized = score_map_np

                # Apply bilateral filter (preserves edges, smooths homogeneous regions)
                score_map_filtered = cv2.bilateralFilter(
                    score_map_normalized,
                    d=self.spatial_filter_d,
                    sigmaColor=self.spatial_filter_sigma_color / 100.0,  # Scale to [0, 1] range
                    sigmaSpace=self.spatial_filter_sigma_space
                )

                # Denormalize back
                if score_max > score_min:
                    score_map_filtered = score_map_filtered * (score_max - score_min) + score_min

                # Convert back to tensor
                score_map_up = torch.from_numpy(score_map_filtered).to(self.device)

            # Get max score
            max_score = float(score_map_up.max())

            # CPU transfer
            score_map_np = score_map_up.cpu().numpy()

            # Validate
            if np.isnan(score_map_np).any() or np.isinf(score_map_np).any():
                score_map_np = np.nan_to_num(score_map_np, nan=0.0, posinf=0.0, neginf=0.0)
                max_score = float(np.max(score_map_np)) if not np.isnan(max_score) else 0.0

            # Track timing
            inference_time = (time.perf_counter() - t_start) * 1000.0
            self.inference_times.append(inference_time)

            if return_timing:
                timing = {
                    "index_search_ms": knn_time_ms,
                    "preprocess_ms": preprocess_ms,
                    "total_inference_ms": inference_time
                }
                return score_map_np, max_score, timing
            else:
                return score_map_np, max_score

    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process frame with motion detection and inference.

        This is the main entry point for real-time inference with motion filtering.

        Args:
            frame: Input frame [H, W, 3] BGR uint8

        Returns:
            Dict with:
                - current_fps: Current FPS estimate
                - motion_active: True if motion detected with hysteresis
                - motion_amount: Fraction of pixels in motion (0.0 - 1.0)
                - max_score: Maximum anomaly score (0.0 if skipped)
                - anomaly_map: Anomaly heatmap [heatmap_size, heatmap_size] (None if skipped)
                - detection_skipped: True if anomaly detection was skipped
                - total_inference_ms: Total inference time in ms
                - index_search_ms: k-NN search time in ms
                - preprocess_ms: Preprocessing time in ms
        """
        # Track FPS
        current_time = time.time()
        self.frame_times.append(current_time)

        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.current_fps = len(self.frame_times) / time_diff if time_diff > 0 else self.current_fps

            # Update motion filter FPS
            if self.motion_filter is not None:
                self.motion_filter.update_fps(self.current_fps)

        # Initialize result
        result = {
            'current_fps': self.current_fps,
            'motion_active': False,
            'motion_amount': 0.0,
            'max_score': 0.0,
            'anomaly_map': None,
            'detection_skipped': False,
            'total_inference_ms': 0.0,
            'index_search_ms': 0.0,
            'preprocess_ms': 0.0,
        }

        # Motion Detection (if enabled)
        motion_active = False
        motion_amount = 0.0
        if self.motion_filter is not None:
            motion_active, motion_amount = self.motion_filter.update(frame)
            result['motion_active'] = motion_active
            result['motion_amount'] = motion_amount

            # Skip inference if motion active
            if motion_active:
                result['detection_skipped'] = True
                return result

        # Run full inference
        if self.is_ready:
            try:
                anomaly_map, max_score, timing = self.predict(frame, return_timing=True)
                result['max_score'] = max_score
                result['anomaly_map'] = anomaly_map
                result['total_inference_ms'] = timing.get('total_inference_ms', 0.0)
                result['index_search_ms'] = timing.get('index_search_ms', 0.0)
                result['preprocess_ms'] = timing.get('preprocess_ms', 0.0)

            except Exception as e:
                print(f"[ERROR] Anomaly detection failed: {e}")
                result['max_score'] = 0.0
                result['anomaly_map'] = None

        return result

    def get_avg_fps(self) -> float:
        """Get average FPS from recent inferences."""
        if not self.inference_times:
            return 0.0

        avg_time_ms = np.mean(self.inference_times[-30:])
        return 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    def create_overlay(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        threshold: float = 0.5,
        max_score: float = 0.0,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create heatmap overlay on image with Gaussian smoothing.

        Args:
            image: Original image [H, W, 3]
            anomaly_map: Anomaly map [heatmap_size, heatmap_size]
            threshold: Anomaly threshold (not used)
            max_score: Maximum score (not used)
            colormap: OpenCV colormap (COLORMAP_JET)
            alpha: Overlay transparency (0.0 = invisible, 1.0 = opaque)

        Returns:
            overlay: Image with heatmap overlay [H, W, 3]
        """
        # Validate inputs
        if image is None or anomaly_map is None:
            return image if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = image.shape[:2]

        # Clean NaN/Inf
        if np.isnan(anomaly_map).any() or np.isinf(anomaly_map).any():
            anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply Gaussian smoothing
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=2.0)

        # Resize to match image
        if anomaly_map.shape[:2] != (h, w):
            anomaly_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            anomaly_resized = anomaly_map

        # Normalize and colorize
        anom_uint8 = np.clip(anomaly_resized * 255, 0, 255).astype(np.uint8)
        anom_colored = cv2.applyColorMap(anom_uint8, colormap)
        anom_colored = cv2.cvtColor(anom_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(image, 1.0 - alpha, anom_colored, alpha, 0)

        return overlay


# Example usage / Testing
if __name__ == "__main__":
    print("Live Anomaly Detector Test")
    print("=" * 50)

    # Example: Initialize detector with reference images
    reference_path = Path("path/to/your/dataset")  # Adjust this!

    detector = LiveAnomalyDetector(
        model_name="facebook/dinov3-vits16-pretrain",  # Fast model
        reference_path=reference_path,
        shots=20,
        knn_k=5,
        use_faiss=True,
        use_coreset=True,
        coreset_ratio=0.1
    )

    # Example: Process a test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    anomaly_map, score = detector.predict(test_img)

    print(f"Anomaly score: {score:.4f}")
    print(f"Avg FPS: {detector.get_avg_fps():.1f}")
