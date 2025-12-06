"""
main_pipeline.py
Unified entrypoint to run:
 - quick synthetic test (no KITTI, no OverlapNet required)
 - full KITTI run (requires KITTI velodyne files, OverlapNet repo & checkpoint, optional TEASER++)
Run:
  python3 code/main_pipeline.py --mode synthetic
  python3 code/main_pipeline.py --mode kitti --bindir /path/to/velodyne --query 150 --overlapnet /path/to/OverlapNet --checkpoint /path/to/checkpoint.pth
"""
import argparse, os, numpy as np
from sklearn.neighbors import KDTree

# reuse earlier synthetic pipeline code if present
# We'll implement a small synthetic generator inline for the synthetic test
def generate_synthetic_map(n_scans=5, points_per_scan=2000, spread=0.6, seed=42):
    import numpy as np
    np.random.seed(seed)
    scans=[]
    for i in range(n_scans):
        x = np.random.uniform(-8,8,points_per_scan)
        y = np.random.uniform(-4,4,points_per_scan)
        z = 0.2*np.sin(0.5*x) + 0.1*np.cos(0.3*y) + 0.02*np.random.randn(points_per_scan)
        pts = np.vstack([x + i*spread, y, z]).T
        scans.append(pts)
    merged = np.vstack(scans)
    return merged, scans

def make_query_from_map(merged_map, overlap_fraction=0.08, new_points=1500, seed=43):
    import numpy as np
    np.random.seed(seed)
    n_map = merged_map.shape[0]
    n_overlap = int(n_map * overlap_fraction)
    xs = merged_map[:,0]
    slab_center = np.percentile(xs, 50) + np.random.uniform(-1.0,1.0)
    slab_idx = np.where((xs > slab_center-4.0) & (xs < slab_center+4.0))[0]
    if len(slab_idx) < n_overlap:
        choice = np.random.choice(n_map, n_overlap, replace=False)
    else:
        choice = np.random.choice(slab_idx, n_overlap, replace=False)
    overlap_pts = merged_map[choice]
    moved = overlap_pts.copy()
    k = int(0.05 * moved.shape[0])
    if k>0:
        inds = np.random.choice(moved.shape[0], k, replace=False)
        moved[inds] += np.array([0.5, 0.5, 0.05])
    new_pts = np.random.uniform(low=[-2,-6,-0.1], high=[6,6,0.5], size=(new_points,3))
    query = np.vstack([moved, new_pts])
    ang = np.deg2rad(np.random.uniform(-20,20))
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang),  np.cos(ang), 0],
                  [0,0,1]])
    t = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), 0.0])
    query = (R @ query.T).T + t
    return query

def run_synthetic(out_dir="out_synth", use_gate=False, overlapnet_root=None, checkpoint=None):
    os.makedirs(out_dir, exist_ok=True)
    merged_map, scans = generate_synthetic_map(n_scans=5, points_per_scan=2000)
    query = make_query_from_map(merged_map, overlap_fraction=0.08, new_points=1500)
    np.save(os.path.join(out_dir,"map.npy"), merged_map)
    np.save(os.path.join(out_dir,"query.npy"), query)
    # optionally use OverlapNet gating to select subset of scans
    selected_map = merged_map
    if use_gate and overlapnet_root is not None:
        from overlapnet_infer import OverlapNetInfer
        infer = OverlapNetInfer(overlapnet_root, checkpoint=checkpoint, use_pretrained=True)
        scores = [float(infer.score_pair(query, s)) for s in scans]
        topk = np.argsort(scores)[-3:][::-1]
        selected = [scans[i] for i in topk]
        selected_map = np.vstack(selected)
    # robust registration: try TEASER or trICP
    from robust_registration import teaser_registration, tr_icp
    T = teaser_registration(query, selected_map)
    if T is None:
        T, inlier_mask, dists_all = tr_icp(query, selected_map, trim_frac=0.6, max_iter=80, verbose=True)
        R = T[:3,:3]; t = T[:3,3]
        reg = (R @ query.T).T + t
    else:
        R = T[:3,:3]; t = T[:3,3]
        reg = (R @ query.T).T + t
    tree = KDTree(selected_map)
    dists, _ = tree.query(reg, k=1)
    mask = dists.ravel() < 0.35
    np.save(os.path.join(out_dir,"registered_query.npy"), reg)
    np.save(os.path.join(out_dir,"overlap_mask.npy"), mask)
    meta = {"mean_nn": float(dists.mean()), "inlier_frac": float(mask.mean())}
    with open(os.path.join(out_dir,"meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print("Synthetic run finished. Saved outputs to", out_dir)
    return meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synthetic","kitti"], default="synthetic")
    p.add_argument("--out", default="out_run")
    # kitti args
    p.add_argument("--bindir", help="KITTI velodyne bin folder")
    p.add_argument("--query", type=int, help="frame index for KITTI")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--overlapnet_root", default=os.environ.get("OVERLAPNET_ROOT", None))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no_teaser", action="store_true")
    args = p.parse_args()
    if args.mode == "synthetic":
        run_synthetic(out_dir=args.out, use_gate=(args.overlapnet_root is not None), overlapnet_root=args.overlapnet_root, checkpoint=args.checkpoint)
    else:
        # import run_on_kitti (this file in the same folder)
        from run_on_kitti import run_sequence
        run_sequence(args.bindir, args.query, window=args.window, topk_gate=args.topk, overlapnet_root=args.overlapnet_root, checkpoint=args.checkpoint, outdir=args.out, use_teaser=(not args.no_teaser))

if __name__ == "__main__":
    main()

"""
export OVERLAPNET_ROOT=~/projects/OverlapNet

Run:
  python3 code/main_pipeline.py --mode synthetic
  python3 code/main_pipeline.py --mode kitti --bindir /path/to/velodyne --query 150 --overlapnet /path/to/OverlapNet --checkpoint /path/to/checkpoint.pth
"""