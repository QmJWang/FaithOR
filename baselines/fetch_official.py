"""Fetch pinned official baseline repositories without redistributing their code."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(Path(__file__).with_name("official.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for item in sources:
        dst = args.out / item["name"]
        if dst.exists():
            raise FileExistsError(dst)
        subprocess.run(["git", "clone", item["url"], str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "checkout", "--detach", item["commit"]], check=True)


if __name__ == "__main__":
    main()
