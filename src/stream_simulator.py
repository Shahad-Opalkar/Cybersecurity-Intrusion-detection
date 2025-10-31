# src/stream_simulator.py
import time
import argparse
import pandas as pd
import os
from src.utils.dataset import load_nslkdd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

def simulate(output_csv="stream_out.csv", delay=0.5, max_rows=500, shuffle=True):
    df = load_nslkdd(TEST_FILE)
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(max_rows)
    out_path = os.path.join(BASE_DIR, "..", output_csv)
    df.iloc[:0].to_csv(out_path, index=False)
    for idx, row in df.iterrows():
        row.to_frame().T.to_csv(out_path, mode="a", header=False, index=False)
        print(f"Emitted row {idx}")
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="stream_out.csv")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    args = parser.parse_args()
    simulate(output_csv=args.out, delay=args.delay, max_rows=args.max, shuffle=args.shuffle)
