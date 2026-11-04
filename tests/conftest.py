import os

# Tests measure the simulation, never the renderer. With GL disabled, any code
# path that tries to create a mujoco.Renderer fails loudly instead of quietly
# needing a display. Set before mujoco is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "disable")
