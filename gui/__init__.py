"""
GUI package for Anomaly Detection App.

Provides modular GUI components, dialogs, threads, handlers, and utilities.
"""

# Threads
from .threads import TrainingThread, InferenceThread, CameraThread, USBCameraThread

# Dialogs
from .dialogs import (
    ProjectSelectionDialog,
    ProjectConfigDialog,
    ModelSelectionDialog,
    RetrainingConfigDialog,
    CameraTypeSelectionDialog,
)

# Utilities & Effects
from .utils import image_to_numpy_rgb, numpy_to_qimage
from .visual_effects import VisualEffectsMixin
from .ui_builder import UIBuilder

# Handlers
from .handlers import (
    CameraHandler,
    DetectionHandler,
    TrainingHandler,
    ProjectHandler,
    CaptureHandler
)

__all__ = [
    # Threads
    'TrainingThread',
    'InferenceThread',
    'CameraThread',
    'USBCameraThread',
    # Dialogs
    'ProjectSelectionDialog',
    'ProjectConfigDialog',
    'ModelSelectionDialog',
    'RetrainingConfigDialog',
    'CameraTypeSelectionDialog',
    # Utilities & Effects
    'image_to_numpy_rgb',
    'numpy_to_qimage',
    'VisualEffectsMixin',
    'UIBuilder',
    # Handlers
    'CameraHandler',
    'DetectionHandler',
    'TrainingHandler',
    'ProjectHandler',
    'CaptureHandler',
]
