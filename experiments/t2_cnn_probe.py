"""Leaderboard probes of the convolutional ongoing model (tfb.t2_cnn): Task 2 rebuilt at given weights.

p = (1 - pub - cnn) ours + pub public + cnn cnn over every link, then the expected-IoU set; Tasks 1 and 4 copied.

    TFB_CNN_DEVICE=mps python experiments/t2_cnn_probe.py --out v43 --pub 0.25 --cnn 0.5 [--base v42]
"""
import argparse
import sys

from tfb import pipeline_final, t2_cnn, t2_public

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--pub", type=float, required=True)
ap.add_argument("--cnn", type=float, required=True)
ap.add_argument("--base", default="v42")
ap.add_argument("--off", type=float, default=None, help="public-model scale off our candidate cells (default: --pub)")
args = ap.parse_args()
t2_public.WEIGHT, t2_cnn.WEIGHT, t2_public.OFF_SCALE = args.pub, args.cnn, args.off
sys.argv = ["pipeline_final", "--out", args.out, "--tasks", "2", "--base", args.base]
pipeline_final.main()
