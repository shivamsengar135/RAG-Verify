"""Download the official Self-RAG checkpoint with parallel HTTP range requests and verify SHA-256.

Needed because single-connection downloads from the HF CDN ran at ~0.2 MB/s on this network.
Assembles a self-contained model folder at data/models/selfrag_llama2_7b_official/ (main branch files only).

Usage: .venv\\Scripts\\python.exe code\\fetch_selfrag.py [--parts 16]
"""
import argparse
import hashlib
import json
import shutil
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

from common import ROOT

REPO = "selfrag/selfrag_llama2_7b"
OUT = ROOT / "data" / "models" / "selfrag_llama2_7b_official"
CTX = ssl.create_default_context(cafile=certifi.where())
SMALL = ["config.json", "generation_config.json", "added_tokens.json", "special_tokens_map.json",
         "tokenizer_config.json", "tokenizer.model", "pytorch_model.bin.index.json"]


def url(path):
    return f"https://huggingface.co/{REPO}/resolve/main/{path}"


def get_part(path, start, end, part_file):
    """Download bytes [start, end] into part_file, resuming if it already holds some of them."""
    want = end - start + 1
    for attempt in range(50):
        have = part_file.stat().st_size if part_file.exists() else 0
        if have >= want:
            return
        req = urllib.request.Request(url(path), headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=60) as r, part_file.open("ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:  # network hiccup: retry from where we stopped
            print(f"  part {part_file.name} retry {attempt}: {e!r}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"part {part_file} did not complete")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch_large(path, size, oid, parts, snapshot_copy=None):
    dest = OUT / path
    if dest.exists() and dest.stat().st_size == size:
        print(f"{path}: present")
    elif snapshot_copy and snapshot_copy.exists() and snapshot_copy.stat().st_size == size:
        shutil.copyfile(snapshot_copy, dest)
        print(f"{path}: copied from HF cache")
    else:
        tmp = OUT / (path + ".parts")
        tmp.mkdir(exist_ok=True)
        step = -(-size // parts)
        ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(parts)]
        t0 = time.time()
        with ThreadPoolExecutor(parts) as ex:
            futs = [ex.submit(get_part, path, s, e, tmp / f"{i:03d}") for i, s, e in ranges]
            while not all(f.done() for f in futs):
                got = sum(p.stat().st_size for p in tmp.iterdir())
                rate = got / max(time.time() - t0, 1) / 2**20
                eta = (size - got) / max(rate * 2**20, 1) / 60
                print(f"  {path}: {got / 2**30:.2f}/{size / 2**30:.2f} GiB  {rate:.1f} MB/s  eta {eta:.0f} min",
                      flush=True)
                time.sleep(30)
            for f in futs:
                f.result()
        with dest.open("wb") as out:
            for i, _, _ in ranges:
                with (tmp / f"{i:03d}").open("rb") as f:
                    shutil.copyfileobj(f, out, 1 << 24)
        shutil.rmtree(tmp)
    digest = sha256(dest)
    ok = digest == oid
    print(f"{path}: sha256 {'OK' if ok else 'MISMATCH ' + digest}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tree = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/main", context=CTX).read())
    lfs = {t["path"]: (t["size"], t["lfs"]["oid"]) for t in tree if t.get("lfs")}
    for name in SMALL:
        if not (OUT / name).exists():
            (OUT / name).write_bytes(urllib.request.urlopen(url(name), context=CTX, timeout=120).read())
    snap = Path.home() / ".cache" / "huggingface" / "hub" / "models--selfrag--selfrag_llama2_7b" / "snapshots"
    for path, (size, oid) in sorted(lfs.items()):
        if path.endswith(".bin"):
            cached = next(snap.glob(f"*/{path}"), None) if snap.exists() else None
            fetch_large(path, size, oid, args.parts, cached)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
