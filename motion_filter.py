"""
Motion Detection with Hysteresis Filtering
==========================================

FPS-adaptive motion detection using OpenCV BackgroundSubtractorMOG2
with dual-threshold hysteresis to prevent false triggers.

Usage:
    filter = MotionHysteresisFilter(
        motion_high_threshold=0.05,  # 5% motion to trigger
        motion_low_threshold=0.01,   # 1% motion to release
        stabilization_time_sec=2.0,  # Wait 2s after motion stops
        learning_time_sec=10.0,      # Learn background over 10s
        estimated_fps=10.0           # Camera FPS
    )

    # Process each frame
    motion_active, motion_amount = filter.update(frame)

    if motion_active:
        print("Motion detected - skipping anomaly detection")
    else:
        # Safe to run anomaly detection
        pass
"""

import cv2
import numpy as np
from typing import Tuple


class MotionHysteresisFilter:
    """
    FPS-adaptive motion detection with dual-threshold hysteresis.

    Uses OpenCV's BackgroundSubtractorMOG2 for robust motion detection
    and hysteresis to prevent flicker at threshold boundaries.

    Attributes:
        motion_active: Current motion state (True = motion detected)
        estimated_fps: Current frame rate estimate
    """

    def __init__(
        self,
        motion_high_threshold: float = 0.05,
        motion_low_threshold: float = 0.01,
        stabilization_time_sec: float = 2.0,
        learning_time_sec: float = 10.0,
        estimated_fps: float = 10.0
    ):
        """
        Initialize motion hysteresis filter.

        Args:
            motion_high_threshold: Fraction of pixels (0-1) to trigger motion (e.g., 0.05 = 5%)
            motion_low_threshold: Fraction of pixels (0-1) to release motion (e.g., 0.01 = 1%)
            stabilization_time_sec: Seconds to wait after motion stops before allowing detection
            learning_time_sec: Seconds of frames for background model learning
            estimated_fps: Initial FPS estimate for frame count calculations
        """
        self.high_threshold = motion_high_threshold
        self.low_threshold = motion_low_threshold
        self.stabilization_time_sec = stabilization_time_sec
        self.learning_time_sec = learning_time_sec
        self.estimated_fps = estimated_fps

        # Calculate frame-based parameters (FPS-adaptive)
        self._recalculate_frame_params()

        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.learning_frames,
            varThreshold=16,           # Threshold for pixel classification
            detectShadows=True         # Detect and ignore shadows
        )

        # State tracking
        self.motion_active = False
        self.frames_since_motion_stop = 0
        self.last_motion_amount = 0.0

    def _recalculate_frame_params(self):
        """Recalculate frame-based parameters from FPS."""
        self.stabilization_frames = max(1, int(self.stabilization_time_sec * self.estimated_fps))
        self.learning_frames = max(10, int(self.learning_time_sec * self.estimated_fps))

    def update_fps(self, new_fps: float):
        """
        Update FPS estimate and recalculate frame counts.

        Args:
            new_fps: New FPS estimate
        """
        if new_fps <= 0:
            return  # Invalid FPS

        self.estimated_fps = new_fps
        self._recalculate_frame_params()

        # Update BackgroundSubtractor history
        # Note: MOG2 doesn't have a direct setter, but new frames will adapt
        # We can't change history dynamically, but the filter will adapt over time

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Process frame and update motion state.

        Args:
            frame: Input frame [H, W, 3] BGR uint8

        Returns:
            (motion_active, motion_amount):
                - motion_active: True if motion detected with hysteresis
                - motion_amount: Fraction of pixels in motion (0.0 - 1.0)
        """
        # Convert to grayscale for faster processing
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_gray)

        # Calculate motion amount (fraction of foreground pixels)
        total_pixels = fg_mask.size
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_amount = motion_pixels / total_pixels if total_pixels > 0 else 0.0

        self.last_motion_amount = motion_amount

        # Dual-threshold hysteresis logic
        if not self.motion_active:
            # Not in motion state - check if HIGH threshold exceeded
            if motion_amount > self.high_threshold:
                self.motion_active = True
                self.frames_since_motion_stop = 0
                # print(f"[MotionFilter] Motion triggered: {motion_amount:.3f} > {self.high_threshold:.3f}")
        else:
            # In motion state - check if LOW threshold undershot
            if motion_amount < self.low_threshold:
                # Motion stopped, but start stabilization countdown
                self.frames_since_motion_stop += 1

                if self.frames_since_motion_stop >= self.stabilization_frames:
                    self.motion_active = False
                    self.frames_since_motion_stop = 0
                    # print(f"[MotionFilter] Motion released after {self.stabilization_frames} frames")
            else:
                # Motion still above LOW threshold - reset counter
                self.frames_since_motion_stop = 0

        return self.motion_active, motion_amount

    def reset(self):
        """Reset motion state and background model."""
        self.motion_active = False
        self.frames_since_motion_stop = 0
        self.last_motion_amount = 0.0

        # Recreate background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.learning_frames,
            varThreshold=16,
            detectShadows=True
        )

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get current foreground mask (for visualization).

        Args:
            frame: Input frame [H, W, 3] BGR uint8

        Returns:
            Foreground mask [H, W] uint8 (0 or 255)
        """
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        return self.bg_subtractor.apply(frame_gray)

    def get_status_text(self) -> str:
        """
        Get human-readable status text.

        Returns:
            Status string with current state
        """
        if self.motion_active:
            if self.frames_since_motion_stop > 0:
                return (f"MOTION ACTIVE (stabilizing {self.frames_since_motion_stop}/{self.stabilization_frames}): "
                       f"{self.last_motion_amount:.1%}")
            else:
                return f"MOTION ACTIVE: {self.last_motion_amount:.1%}"
        else:
            return f"STATIC: {self.last_motion_amount:.1%}"

    def __repr__(self) -> str:
        return (f"MotionHysteresisFilter("
                f"high={self.high_threshold:.3f}, "
                f"low={self.low_threshold:.3f}, "
                f"stab={self.stabilization_time_sec}s @ {self.estimated_fps:.1f}fps, "
                f"state={'MOTION' if self.motion_active else 'STATIC'})")


if __name__ == "__main__":
    """Test motion detection with webcam."""
    import time

    print("Testing MotionHysteresisFilter...")
    print("Move in front of camera to trigger motion detection")
    print("Press 'q' to quit")

    # Initialize filter with test parameters
    motion_filter = MotionHysteresisFilter(
        motion_high_threshold=0.05,
        motion_low_threshold=0.01,
        stabilization_time_sec=2.0,
        learning_time_sec=5.0,
        estimated_fps=10.0
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    fps_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track FPS
        fps_times.append(time.time())
        if len(fps_times) > 30:
            fps_times.pop(0)

        if len(fps_times) >= 2:
            fps = len(fps_times) / (fps_times[-1] - fps_times[0])
            motion_filter.update_fps(fps)
        else:
            fps = 0.0

        # Update motion detection
        motion_active, motion_amount = motion_filter.update(frame)

        # Get foreground mask for visualization
        fg_mask = motion_filter.get_foreground_mask(frame)

        # Draw status on frame
        status_color = (0, 0, 255) if motion_active else (0, 255, 0)
        status_text = motion_filter.get_status_text()

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame and mask
        cv2.imshow("Motion Detection", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
