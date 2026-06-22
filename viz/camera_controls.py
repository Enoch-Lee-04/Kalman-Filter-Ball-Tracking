"""OpenCV mouse/keyboard bindings for the orbit camera."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2

from sim.orbit_camera import OrbitCamera


@dataclass
class CameraControls:
    orbit_camera: OrbitCamera
    window_name: str
    is_orbit_active: Callable[[], bool] = field(default=lambda: True)
    dragging: bool = False
    last_xy: tuple[int, int] = field(default=(0, 0))
    azimuth_sensitivity: float = 0.35
    elevation_sensitivity: float = 0.25
    zoom_step: float = 0.35

    def attach(self) -> None:
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def handle_key(self, key: int) -> bool:
        """Apply keyboard shortcuts. Returns True if the key was consumed."""
        if not self.is_orbit_active():
            return False
        if key in (ord("+"), ord("=")):
            self.orbit_camera.zoom(self.zoom_step * self.orbit_camera.distance * 0.08)
            return True
        if key in (ord("-"), ord("_")):
            self.orbit_camera.zoom(-self.zoom_step * self.orbit_camera.distance * 0.08)
            return True
        if key in (ord("r"), ord("R")):
            self.orbit_camera.reset()
            return True
        return False

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if not self.is_orbit_active():
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            self.last_xy = (x, y)
            self.orbit_camera.orbit(
                dx * self.azimuth_sensitivity,
                dy * self.elevation_sensitivity,
            )
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Windows: flags >> 16 is signed delta in multiples of 120.
            delta = (flags >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000
            step = self.orbit_camera.distance * 0.002 * delta
            self.orbit_camera.zoom(step)
