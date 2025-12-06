"""
overlapnet_infer.py
Wrapper to call PRBonn/OverlapNet inference.

Usage:
    from overlapnet_infer import OverlapNetInfer
    infer = OverlapNetInfer(overlapnet_root="/path/to/OverlapNet", checkpoint="/path/to/checkpoint.pth")
    score = infer.score_pair(pc_query, pc_map_keyframe)
Notes:
 - This wrapper tries to import an OverlapNet Python API if available in the repo.
 - If import fails, it will call the repo's inference script via subprocess (CLI) if available.
 - You MUST follow PRBonn/OverlapNet README to download dataset preprocessing code and checkpoint.
References: https://github.com/PRBonn/OverlapNet
"""
import os, tempfile, subprocess, numpy as np, sys, json

class OverlapNetInfer:
    def __init__(self, overlapnet_root, checkpoint=None, use_pretrained=True, device="cpu"):
        self.root = os.path.abspath(overlapnet_root) if overlapnet_root else None
        self.checkpoint = checkpoint
        self.use_pretrained = use_pretrained
        self.device = device
        # Try to import repo code (user must `python setup.py install` or add to PYTHONPATH)
        self.api_available = False
        if self.root:
            sys.path.insert(0, self.root)
            try:
                # Many forked OverlapNet repos expose an Infer or OverlapNet inference API.
                # Try common import names (best-effort).
                from infer import Infer  # PRBonn style infer class (may differ by version)
                self.Infer = Infer
                self.api_available = True
            except Exception:
                # fallback: no importable API
                self.api_available = False

    def _pc_to_range_image_tempfile(self, pc):
        # Save a temporary npy for the repo to read if CLI-mode is used
        fd, path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(path, pc.astype(np.float32))
        return path

    def score_pair(self, pcA, pcB):
        """
        pcA, pcB: Nx3 numpy arrays
        returns: float overlap score in [0,1]
        """
        if self.use_pretrained and self.api_available:
            # instantiate repo's Infer -- the constructor signature may vary across versions of OverlapNet.
            try:
                inf = self.Infer(self.root, checkpoint=self.checkpoint, device=self.device)
                score = inf.predict_overlap(pcA, pcB)
                return float(score)
            except Exception as e:
                print("OverlapNet API call failed:", e)
                # fall through to CLI fallback

        # CLI fallback: try to call `python infer.py --pc1 <file> --pc2 <file> ...`
        # This is a generic pattern: if the repo provides a script taking .npy inputs, it will work.
        # You may need to adapt the command below to the OverlapNet repo's actual inference script.
        if self.root is None:
            raise RuntimeError("overlapnet_root is not set and API import failed.")
        # Attempt to find an inference script (common names)
        possible_scripts = ["infer.py", "run_infer.py", "infer_overlap.py"]
        script = None
        for s in possible_scripts:
            path = os.path.join(self.root, s)
            if os.path.exists(path):
                script = path
                break
        if script is None:
            raise RuntimeError(f"Could not find OverlapNet inference script in {self.root}. Check the repo and README.")

        fA = self._pc_to_range_image_tempfile(pcA)
        fB = self._pc_to_range_image_tempfile(pcB)
        # construct CLI command (user may need to change flags based on repo's CLI)
        cmd = ["python3", script, "--pc1", fA, "--pc2", fB]
        if self.checkpoint:
            cmd += ["--checkpoint", self.checkpoint]
        try:
            out = subprocess.check_output(cmd, cwd=self.root, stderr=subprocess.STDOUT, timeout=60)
            txt = out.decode("utf-8")
            # repo CLI may print JSON or a value; try to parse last float in output
            for line in reversed(txt.strip().splitlines()):
                try:
                    val = float(line.strip())
                    # cleanup
                    os.remove(fA); os.remove(fB)
                    return float(val)
                except Exception:
                    continue
            # fallback: return 0
            os.remove(fA); os.remove(fB)
            return 0.0
        except subprocess.CalledProcessError as e:
            print("OverlapNet CLI failed:", e.output.decode("utf-8"))
            os.remove(fA); os.remove(fB)
            return 0.0
