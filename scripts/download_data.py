"""Fetch every downloadable dataset into data/raw/<key>.jsonl.

    python scripts/download_data.py                 # everything reachable
    python scripts/download_data.py --lang bn       # one language
    python scripts/download_data.py --keys popqa dolly
    python scripts/download_data.py --limit 2000    # cap rows per set (fast probe)

Datasets that need manual acquisition (BLUCK, BEnQA, BanglaMedQA) are reported
as SKIPPED with the reason -- see DATASETS.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collapse.data.loaders import load_dataset_key
from collapse.data.registry import REGISTRY, by_lang
from collapse.utils import LOG, write_jsonl

OUT = Path("data/raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "bn"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per dataset (default: all)")
    ap.add_argument("--force", action="store_true", help="re-download existing")
    args = ap.parse_args()

    if args.keys:
        specs = [REGISTRY[k] for k in args.keys]
    elif args.lang:
        specs = by_lang(args.lang)
    else:
        specs = list(REGISTRY.values())

    OUT.mkdir(parents=True, exist_ok=True)
    ok, skipped, failed = [], [], []

    for spec in specs:
        dest = OUT / f"{spec.key}.jsonl"
        if dest.exists() and not args.force:
            LOG.info("exists, skipping: %s", dest)
            ok.append((spec.key, "cached"))
            continue

        if spec.hf_id.startswith("LOCAL:"):
            path = spec.hf_id[len("LOCAL:"):]
            if Path(path).exists():
                ok.append((spec.key, "local file present"))
            else:
                skipped.append((spec.key, f"MANUAL: needs {path} -- {spec.note[:120]}"))
            continue

        t0 = time.time()
        try:
            rows = load_dataset_key(spec.key, limit=args.limit)
            n = write_jsonl(dest, rows)
            ok.append((spec.key, f"{n} rows in {time.time() - t0:.0f}s"))
        except Exception as e:
            failed.append((spec.key, spec.hf_id, str(e).split("\n")[0][:150]))

    print("\n" + "=" * 78)
    print("DOWNLOADED")
    for k, d in ok:
        print(f"  OK       {k:<16} {d}")
    print("\nNEEDS MANUAL ACQUISITION  (see DATASETS.md)")
    for k, d in skipped:
        print(f"  MANUAL   {k:<16} {d}")
    if failed:
        print("\nFAILED")
        for k, hf, e in failed:
            print(f"  FAIL     {k:<16} ({hf})\n           {e}")
    print("=" * 78)
    print(f"{len(ok)} ok / {len(skipped)} manual / {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
