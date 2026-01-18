"""Application state enum."""
from enum import Enum


class AppState(Enum):
    """Application states."""
    PROJECT_SELECTION = "project_selection"

    # Text overlay intros (3s black screen with white text)
    SHOW_STEP_1_INTRO = "show_step_1_intro"  # Background intro
    SHOW_STEP_2_INTRO = "show_step_2_intro"  # Good images intro
    SHOW_STEP_3_INTRO = "show_step_3_intro"  # Defect images intro

    # Capture states
    CAPTURE_BACKGROUND = "capture_background"  # STEP 1: Capture 5 background images (no object)
    CAPTURE_GOOD = "capture_good"              # STEP 2: Capture good images (train+test merged)
    CAPTURE_TEST_DEFECT = "capture_test_defect"  # STEP 3: Capture defect images

    # Training and detection
    TRAINING = "training"
    LIVE_DETECTION = "live_detection"
