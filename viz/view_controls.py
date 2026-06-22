"""Toggle between full-screen orbit and robot camera views."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewState:
    mode: str = "orbit"  # "orbit" | "robot"

    @property
    def is_orbit(self) -> bool:
        return self.mode == "orbit"

    @property
    def is_robot(self) -> bool:
        return self.mode == "robot"

    def toggle(self) -> str:
        self.mode = "robot" if self.is_orbit else "orbit"
        return self.mode


def handle_view_key(key: int, state: ViewState) -> bool:
    """Tab switches view. Returns True if consumed."""
    if key == 9:  # Tab
        state.toggle()
        return True
    return False
