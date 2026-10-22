"""``python -m mygene_prompts build-cache [--all | --symbols PATH]``.

Subcommands:

* ``build-cache --all`` — fetch every gene symbol in
  ``ExperimentalDesign/data/achilles.h5`` (~17,500 genes).
* ``build-cache --symbols FILE`` — one symbol per line.  Use ``-`` for
  stdin.

Idempotent: re-running only fetches symbols not already in the cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from mygene_prompts.cache import build_full_cache, default_cache_path
from mygene_prompts.client import MyGeneClient


def _gene_universe_from_achilles() -> list[str]:
    """Return the gene-symbol list from the Achilles HDF5 file."""
    import os
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # mas-hackathon/
    ach = repo_root / "ExperimentalDesign" / "data" / "achilles.h5"
    if not ach.exists():
        raise FileNotFoundError(
            f"Cannot locate Achilles HDF5 at {ach}; pass --symbols instead."
        )
    with h5py.File(str(ach), "r") as f:
        rownames = f["rownames"].asstr()[:]
    return [str(s) for s in rownames]


def _read_symbols_file(path: str) -> list[str]:
    if path == "-":
        fh = sys.stdin
        text = fh.read()
    else:
        text = Path(path).read_text()
    syms = [line.strip() for line in text.splitlines()]
    return [s for s in syms if s and not s.startswith("#")]


def cmd_build_cache(args: argparse.Namespace) -> int:
    if args.all and args.symbols:
        print("error: --all and --symbols are mutually exclusive", file=sys.stderr)
        return 2
    if not args.all and not args.symbols:
        print("error: pass --all or --symbols PATH", file=sys.stderr)
        return 2

    if args.all:
        symbols = _gene_universe_from_achilles()
        print(f"Loaded {len(symbols)} symbols from Achilles HDF5.")
    else:
        symbols = _read_symbols_file(args.symbols)
        print(f"Loaded {len(symbols)} symbols from {args.symbols}.")

    out_path = Path(args.out) if args.out else default_cache_path()
    print(f"Cache target: {out_path}")

    client = MyGeneClient()
    t0 = time.time()
    cache = build_full_cache(
        symbols,
        path=out_path,
        client=client,
        chunk=args.chunk,
        pause=args.pause,
        resume=not args.fresh,
        progress=True,
    )
    print(
        f"Done in {time.time() - t0:.1f}s.  Cache holds {len(cache)} records "
        f"at {cache.path} ({cache.path.stat().st_size / 1e6:.1f} MB)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mygene_prompts",
        description="Build or refresh the local MyGene.info gene-record cache.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-cache", help="Build or extend the disk cache.")
    src = build.add_mutually_exclusive_group()
    src.add_argument(
        "--all",
        action="store_true",
        help="Fetch every symbol in ExperimentalDesign/data/achilles.h5.",
    )
    src.add_argument(
        "--symbols",
        metavar="PATH",
        help="Path to a file of newline-separated symbols (or '-' for stdin).",
    )
    build.add_argument(
        "--out", metavar="PATH",
        help=f"Cache output path.  Default: {default_cache_path()}",
    )
    build.add_argument(
        "--chunk", type=int, default=1000,
        help="Symbols per POST request (MyGene allows up to 1000).",
    )
    build.add_argument(
        "--pause", type=float, default=0.5,
        help="Seconds to sleep between chunks (politeness).",
    )
    build.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing cache file; refetch every symbol.",
    )
    build.set_defaults(func=cmd_build_cache)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
