import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# ------------------------ MediaPipe setup & helpers ----------------------------
# mp_drawing = mp.solutions.drawing_utils
# mp_connections = mp.solutions.face_mesh_connections
# mp_face_mesh_module = mp.solutions.face_mesh


# Which MP subsets to use
# USE_SETS = ("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE")
# SETNAME2SET = {
#     "LIPS": mp_connections.FACEMESH_LIPS,
#     "LEFT_EYE": mp_connections.FACEMESH_LEFT_EYE,
#     "RIGHT_EYE": mp_connections.FACEMESH_RIGHT_EYE,
#     "NOSE": mp_connections.FACEMESH_NOSE,
# }

def build_selected_raw_ids_and_labels() -> Tuple[List[int], Dict[int, int]]:
    """
    Returns:
        raw_ids: sorted list of unique MP landmark ids across the selected sets.
        raw_id_to_class: map raw landmark id -> class index (0..len(USE_SETS)-1).
    """
    raw_ids = set()
    raw_id_to_class = {}
    for ci, setname in enumerate(USE_SETS):
        S = SETNAME2SET[setname]
        for (i, j) in S:
            raw_ids.add(i); raw_ids.add(j)
            # Assign first seen class (OK: each id belongs to one region here)
            if i not in raw_id_to_class:
                raw_id_to_class[i] = ci
            if j not in raw_id_to_class:
                raw_id_to_class[j] = ci
    raw_ids = sorted(raw_ids)
    return raw_ids, raw_id_to_class
    
    
RAW_IDS, RAWID2CLS = build_selected_raw_ids_and_labels()
RAWID2IDX = {rid: k for k, rid in enumerate(RAW_IDS)}
N_NODES = len(RAW_IDS)