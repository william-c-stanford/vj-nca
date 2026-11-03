"""Fetch the checked small public-media pilot and prepare normalized video cycles."""

import hashlib
import json
import urllib.request
from pathlib import Path

from vj_nca.data import import_video, make_procedural, merge_datasets

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "datasets/public-loops.json").read_text())
    datasets = []
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for clip in manifest["clips"]:
        path = raw / clip["filename"]
        if not path.exists():
            request = urllib.request.Request(clip["url"], headers={"User-Agent": "vj-nca/0.1"})
            with urllib.request.urlopen(request, timeout=40) as response:
                content = response.read(10_000_001)
            if len(content) > 10_000_000:
                raise ValueError("Public pilot download exceeds 10 MB")
            if hashlib.sha256(content).hexdigest() != clip["sha256"]:
                raise ValueError(
                    "Source changed; review its provenance before replacing the manifest"
                )
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        ds = import_video(
            path,
            clip["name"],
            32,
            32,
            16,
            0,
            clip["duration_seconds"],
            clip["page"],
            clip["license"],
        )
        ds.sources[0].update({key: clip[key] for key in ("author", "license_url", "changes")})
        datasets.append(ds)
    public = merge_datasets(datasets)
    public.save(ROOT / "data/public.npz")
    mixed = merge_datasets([make_procedural(), public])
    mixed.save(ROOT / "data/mixed.npz")
    print(
        json.dumps(
            {
                "public": public.diagnostics(),
                "mixed_names": mixed.names,
                "mixed_fingerprint": mixed.fingerprint(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
