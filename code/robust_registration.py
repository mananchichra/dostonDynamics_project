"""
robust_registration.py
- tr_icp: Trimmed ICP implementation (numpy + KDTree)
- teaser_wrapper: wrapper to call TEASER++ if installed (teaserpp_python)
"""

import numpy as np
from sklearn.neighbors import KDTree

def best_fit_transform(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U,S,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def tr_icp(src, tgt, trim_frac=0.65, max_iter=80, tol=1e-6, verbose=False):
    """
    Trimmed ICP: keep best k correspondences each iter.
    Returns:
      T_total: 4x4 transform mapping src -> tgt
      inlier_mask: boolean mask for src (inlier indices)
      dists_all: final nearest-neighbor distances from registered src to tgt
    """
    src_work = src.copy()
    T_total = np.eye(4)
    prev_err = np.inf
    n = src.shape[0]
    k_keep = max(3, int(n * trim_frac))
    for it in range(max_iter):
        tree = KDTree(tgt)
        dists, idx = tree.query(src_work, k=1)
        dists = dists.ravel(); idx = idx.ravel()
        order = np.argsort(dists)
        keep_idx = order[:k_keep]
        src_keep = src_work[keep_idx]
        tgt_corr = tgt[idx[keep_idx]]
        T = best_fit_transform(src_keep, tgt_corr)
        R = T[:3,:3]; t = T[:3,3]
        src_work = (R @ src_work.T).T + t
        mean_err = float(dists[keep_idx].mean())
        if verbose:
            print(f"trICP iter {it:02d}: mean err {mean_err:.6f}")
        if abs(prev_err - mean_err) < tol:
            if verbose:
                print("trICP converged.")
            break
        prev_err = mean_err
        T_total = T @ T_total
    # final NN distances
    tree = KDTree(tgt)
    dists_all, idx_all = tree.query(src_work, k=1)
    dists_all = dists_all.ravel()
    # final inlier mask: k_keep smallest distances
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[np.argsort(dists_all)[:k_keep]] = True
    return T_total, inlier_mask, dists_all

def teaser_registration(src, tgt, estimate_scale=False):
    """
    Try TEASER++ if installed. Returns 4x4 transform or None.
    Requires: teaserpp_python (Python bindings installed).
    """
    try:
        import teaserpp_python
    except Exception as e:
        # TEASER not installed
        # print("TEASER not installed:", e)
        return None
    # Basic usage pattern (see TEASER++ docs)
    try:
        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.noise_bound = 0.5  # example, tune as needed
        params.estimate_scaling = estimate_scale
        solver = teaserpp_python.RobustRegistrationSolver(params)
        # correspondences (simple NN-based correspondences)
        # here we feed all points; TEASER++ expects correspondences arrays
        # Build correspondences by nearest neighbors (src->tgt)
        from sklearn.neighbors import KDTree
        tree = KDTree(tgt)
        dists, idx = tree.query(src, k=1)
        src_corr = src.T
        tgt_corr = tgt[idx.ravel()].T
        # solve
        solver.solve(src_corr, tgt_corr)
        solution = solver.getSolution()
        R = np.array(solution.rotation).reshape(3,3)
        t = np.array(solution.translation).reshape(3)
        T = np.eye(4)
        T[:3,:3] = R; T[:3,3] = t
        return T
    except Exception as e:
        print("TEASER failed:", e)
        return None
