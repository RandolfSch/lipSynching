import sys, mediapipe as mp
print("mediapipe module path:", getattr(mp, "__file__", "<no __file__>"))
print("mediapipe package:", mp)
try:
    import importlib.metadata as im
    print("mediapipe version:", im.version("mediapipe"))
except Exception as e:
    print("could not read version via metadata:", e)
print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])



print(hasattr(mp, "solutions"))
# Fallback: inspect where drawing_utils lives
import importlib, pkgutil
mods = [m.name for m in pkgutil.iter_modules(mp.__path__)] if hasattr(mp, "__path__") else []
print("Submodules under mediapipe:", mods)
