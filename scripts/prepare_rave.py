"""Download three attributed research-licensed psychedelic loops, verify and decode."""

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import torch

from vj_nca.media import save_contact_sheet
from vj_nca.rave_data import prepare_rave_dataset


def main():
    torch.set_num_threads(2)
    manifest = json.loads(Path("datasets/rave-loops.json").read_text())
    raw = Path("data/raw/electric-sheep")
    raw.mkdir(parents=True, exist_ok=True)
    for size in (128, 512):
        for entry in manifest["clips"]:
            path = raw / entry["file"]
            url = manifest["download_root"] + urllib.parse.quote(entry["file"])
            if not path.exists():
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read(32 * 1024**2 + 1)
                if len(content) > 32 * 1024**2:
                    raise ValueError("Source exceeds the small-clip transfer limit")
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {path}")
                path.write_bytes(content)
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")
        dataset = prepare_rave_dataset(raw, manifest, size)
        target = Path(f"data/rave-{size}.npz")
        dataset.save(target)
        print(
            json.dumps(
                {
                    "path": str(target),
                    "shape": list(dataset.frames.shape),
                    "fingerprint": dataset.fingerprint(),
                    "seams": dataset.diagnostics(),
                }
            ),
            flush=True,
        )
        # One row per source; sampled phases across the complete source loop.
        sheet = torch.nn.functional.interpolate(
            dataset.frames[:, ::8].flatten(0, 1), size=(256, 256), mode="area"
        ).reshape(len(dataset.names), 8, 3, 256, 256)
        sheet = sheet.permute(1, 2, 0, 3, 4).reshape(8, 3, len(dataset.names) * 256, 256)
        save_contact_sheet(target.with_suffix(".png"), sheet)


if __name__ == "__main__":
    main()
