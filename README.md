```
python3 code/main_pipeline.py --mode kitti \
  --bindir /path/to/velodyne_points/data \
  --query 150 \
  --out out_kitti_q150 \
  --overlapnet_root "$OVERLAPNET_ROOT" \
  --checkpoint /path/to/overlapnet_checkpoint.pth
What the flags mean
--mode kitti — run KITTI pipeline.

--bindir — folder with .bin (velodyne) files. Use the path that contains files like 0000000000.bin.

--query 150 — index of the query frame in sorted order (0-based). files[150] will be used as the query.

--out out_kitti_q150 — folder where outputs will be saved (created if needed).

--overlapnet_root — path to the OverlapNet repo root (optional). If not provided, fallback gating (range-image IoU) is used.

--checkpoint — path to the OverlapNet checkpoint (optional; only used if the repo API requires it).

If TEASER++ is installed the pipeline will try TEASER first; otherwise it uses Trimmed-ICP.

```

HOW TO RUN.

