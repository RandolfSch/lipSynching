# graph_build_weighted.py
import numpy as np
import mediapipe as mp
from sklearn.neighbors import NearestNeighbors  # pip install scikit-learn

def build_weighted_adjacency_with_knn(
    canonical_xy: np.ndarray,
    use_sets=("LIPS", "LEFT_EYE", "RIGHT_EYE", "NOSE"),
    k_inter: int = 3,
    w_intra: float = 1.0,
    w_inter: float = 0.3,
):
    """
    Build weighted adjacency for selected MP subsets + k-NN cross-set links.

    Args:
        canonical_xy: (N_raw, 2) array of canonical landmark coords for all 468 MP points.
                      You can pass None if you only want intra-class edges (but we want kNN).
                      If you only have coords for the selected subset, adapt code accordingly.
        use_sets: which MP subsets to include.
        k_inter: number of cross-set neighbors to add per node.
        w_intra: weight for intra-class edges.
        w_inter: weight for inter-class edges.

    Returns:
        raw_ids: list of raw MP landmark ids used, length N
        A_w: (N, N) float32 weighted adjacency
    """
    mp_conn = mp.solutions.face_mesh_connections
    name2set = {
        "LIPS": mp_conn.FACEMESH_LIPS,
        "LEFT_EYE": mp_conn.FACEMESH_LEFT_EYE,
        "RIGHT_EYE": mp_conn.FACEMESH_RIGHT_EYE,
        "NOSE": mp_conn.FACEMESH_NOSE,
        # add brows if desired:
        # "LEFT_EYEBROW": mp_conn.FACEMESH_LEFT_EYEBROW,
        # "RIGHT_EYEBROW": mp_conn.FACEMESH_RIGHT_EYEBROW,
    }
    chosen_sets = [name2set[n] for n in use_sets]

    # 1) Collect unique raw ids
    raw_ids = set()
    set_masks = {}  # raw_id -> set index to know class membership
    for si, S in enumerate(chosen_sets):
        for (i, j) in S:
            raw_ids.add(i); raw_ids.add(j)

    raw_ids = sorted(raw_ids)
    raw_id_to_idx = {rid: k for k, rid in enumerate(raw_ids)}
    N = len(raw_ids)

    # 2) Intra-class edges
    A_w = np.zeros((N, N), dtype=np.float32)
    node_class = np.full(N, -1, dtype=np.int32)

    # Build a reverse map: raw_id -> which chosen_set(s)
    # (rarely an id might appear in multiple sets; we just pick the first.)
    raw_id_to_class = {}
    for si, S in enumerate(chosen_sets):
        for (i, j) in S:
            if i in raw_id_to_idx and i not in raw_id_to_class:
                raw_id_to_class[i] = si
            if j in raw_id_to_idx and j not in raw_id_to_class:
                raw_id_to_class[j] = si

    for rid, idx in raw_id_to_idx.items():
        node_class[idx] = raw_id_to_class.get(rid, -1)

    for si, S in enumerate(chosen_sets):
        for (i, j) in S:
            if i in raw_id_to_idx and j in raw_id_to_idx:
                a = raw_id_to_idx[i]; b = raw_id_to_idx[j]
                A_w[a, b] = max(A_w[a, b], w_intra)
                A_w[b, a] = max(A_w[b, a], w_intra)

    # 3) Inter-class k-NN (based on canonical coordinates)
    if canonical_xy is None:
        # If you don't have canonical coords, you can skip or use graph heuristics.
        return raw_ids, A_w

    # Extract canonical coords for the selected nodes from the full 468 coords
    # If your canonical_xy only contains selected nodes already, adapt here.
    sel_xy = canonical_xy[raw_ids, :]  # shape (N, 2)

    nbrs = NearestNeighbors(n_neighbors=k_inter + 1, algorithm="auto").fit(sel_xy)
    dists, knn_idx = nbrs.kneighbors(sel_xy)  # (N, k+1) includes self at idx 0

    for i in range(N):
        ci = node_class[i]
        cnt = 0
        for j in knn_idx[i, 1:]:  # skip self
            if node_class[j] != ci and node_class[j] != -1:
                # Inter-class neighbor
                A_w[i, j] = max(A_w[i, j], w_inter)
                A_w[j, i] = max(A_w[j, i], w_inter)
                cnt += 1
                if cnt >= k_inter:
                    break

    return raw_ids, A_w
