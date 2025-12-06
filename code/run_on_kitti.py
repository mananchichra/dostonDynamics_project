"""
run_on_kitti.py
Driver for KITTI velodyne scans.

Assumptions:
 - velodyne bins are in folder with filenames like 000000.bin, 000001.bin, ...
 - OverlapNet repo cloned and checkpoint available; set OVERLAPNET_ROOT env var or pass arg
 - TEASER++ python bindings installed (optional) for robust solve
Outputs:
 - For each query frame processed: registered query (.npy), overlap mask (.npy), metadata (.json)
"""

import os, argparse, json, numpy as np
from overlapnet_infer import OverlapNetInfer
from robust_registration import tr_icp, teaser_registration
from sklearn.neighbors import KDTree

def load_velodyne_bin(path):
    # KITTI velodyne bin: float32 Nx4 -> x,y,z,intensity
    data = np.fromfile(path, dtype=np.float32).reshape(-1,4)
    return data[:, :3]

def build_local_map(bindir, frame_indices):
    pcs = []
    for i in frame_indices:
        fn = os.path.join(bindir, f"{i:010d}.bin")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Velodyne file not found: {fn}")
        pcs.append(load_velodyne_bin(fn))
    merged = np.vstack(pcs)
    return merged, pcs

def run_sequence(bindir, query_idx, window=5, topk_gate=3, overlapnet_root=None, checkpoint=None, outdir="out_kitti", use_teaser=True):
    os.makedirs(outdir, exist_ok=True)
    start = max(0, query_idx - window)
    indices = list(range(start, query_idx))
    if len(indices) == 0:
        raise RuntimeError("No prior frames available to build local map.")
    merged_map, kf_pcs = build_local_map(bindir, indices)
    query_pc = load_velodyne_bin(os.path.join(bindir, f"{query_idx:010d}.bin"))
    # Overlap gating using OverlapNet
    infer = None
    if overlapnet_root is not None:
        infer = OverlapNetInfer(overlapnet_root, checkpoint=checkpoint, use_pretrained=True)
    else:
        infer = OverlapNetInfer(overlapnet_root=None, use_pretrained=False)
    # Score all keyframes
    scores = []
    for kf in kf_pcs:
        s = float(infer.score_pair(query_pc, kf))
        scores.append(s)
    scores = np.array(scores)
    # pick top-k keyframes by score
    topk = np.argsort(scores)[-topk_gate:][::-1]
    selected_pcs = [kf_pcs[i] for i in topk]
    merged_selected = np.vstack(selected_pcs)
    # Robust registration: try TEASER first (if installed)
    T_teaser = None
    if use_teaser:
        T_teaser = teaser_registration(query_pc, merged_selected)
    if T_teaser is not None:
        T = T_teaser
        R = T[:3,:3]; t = T[:3,3]
        reg = (R @ query_pc.T).T + t
        # dists
        tree = KDTree(merged_selected)
        dists, _ = tree.query(reg, k=1)
        dists = dists.ravel()
        mask = dists < 0.35
    else:
        # fallback to trICP
        T, inlier_mask, dists_all = tr_icp(query_pc, merged_selected, trim_frac=0.6, max_iter=80, verbose=True)
        R = T[:3,:3]; t = T[:3,3]
        reg = (R @ query_pc.T).T + t
        tree = KDTree(merged_selected)
        dists, _ = tree.query(reg, k=1)
        dists = dists.ravel()
        mask = dists < 0.35
    # save outputs
    np.save(os.path.join(outdir, f"query_{query_idx:010d}_reg.npy"), reg)
    np.save(os.path.join(outdir, f"query_{query_idx:010d}_mask.npy"), mask)
    meta = {
        "query_idx": int(query_idx),
        "map_frames": indices,
        "selected_frames": [int(start + i) for i in topk],
        "scores": scores.tolist(),
        "mean_nn": float(dists.mean()),
        "inlier_fraction": float(mask.mean())
    }
    with open(os.path.join(outdir, f"query_{query_idx:010d}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved outputs to", outdir)
    return meta

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bindir", required=True, help="KITTI velodyne bin files folder")
    p.add_argument("--query", type=int, required=True, help="query frame index")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--overlapnet_root", default=os.environ.get("OVERLAPNET_ROOT", None), help="path to OverlapNet repo")
    p.add_argument("--checkpoint", default=None, help="path to OverlapNet checkpoint")
    p.add_argument("--out", default="out_kitti")
    p.add_argument("--no_teaser", action="store_true", help="disable TEASER use (force trICP)")
    args = p.parse_args()
    run_sequence(args.bindir, args.query, window=args.window, topk_gate=args.topk, overlapnet_root=args.overlapnet_root, checkpoint=args.checkpoint, outdir=args.out, use_teaser=(not args.no_teaser))
