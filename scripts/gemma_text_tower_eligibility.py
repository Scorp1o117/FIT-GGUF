"""M2-Gemma Text-Tower Eligibility Record (GPT hard gate, 2026-09-02).

Checks the converted gemma-4-E4B main GGUF tensor inventory against the
text-tower purity rule: vision tensor count = 0, audio tensor count = 0,
mmproj externalized/absent. Any vision/audio weight in the main GGUF pauses
M2 — the dense Same-top calibration would then attribute bytes to tensors
that never participate in the eval-v1 forward.

Writes an eligibility record JSON next to the calibration experiment results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fit_gguf.gguf import read_gguf_layout  # noqa: E402

VISION_PAT = re.compile(r"^(v\.|mm\.|vision|vitt|vgg)", re.IGNORECASE)
AUDIO_PAT = re.compile(r"^(audio|a\.|aud\.)", re.IGNORECASE)


def classify(name: str) -> str:
    if VISION_PAT.match(name):
        return "vision"
    if AUDIO_PAT.match(name):
        return "audio"
    return "text"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    layout = read_gguf_layout(args.gguf)
    counts = {"vision": [], "audio": [], "text": []}
    for tensor in layout.tensors:
        counts[classify(tensor.name)].append(tensor.name)

    mmproj_files = sorted(p.name for p in args.gguf.parent.glob("*mmproj*"))
    sha = hashlib.sha256()
    with args.gguf.open("rb") as file:
        while chunk := file.read(1 << 22):
            sha.update(chunk)

    eligible = not counts["vision"] and not counts["audio"]
    record = {
        "record": "M2-Gemma Text-Tower Eligibility Record",
        "date": str(date.today()),
        "ruling": "GPT 2026-09-02: main GGUF mixing vision/audio weights pauses M2",
        "gguf": str(args.gguf),
        "sha256": sha.hexdigest(),
        "size_bytes": args.gguf.stat().st_size,
        "tensor_total": len(layout.tensors),
        "text_tensor_count": len(counts["text"]),
        "vision_tensor_count": len(counts["vision"]),
        "audio_tensor_count": len(counts["audio"]),
        "vision_names_head": counts["vision"][:10],
        "audio_names_head": counts["audio"][:10],
        "mmproj_files_in_dir": mmproj_files,
        "text_tensor_names": counts["text"],
        "eligible": eligible,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"eligible={eligible} text={len(counts['text'])} "
          f"vision={len(counts['vision'])} audio={len(counts['audio'])} "
          f"mmproj={mmproj_files}")
    print(f"record: {args.out}")
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
