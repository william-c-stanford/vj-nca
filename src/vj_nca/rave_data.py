from pathlib import Path

from .data import LoopDataset, import_video, merge_datasets


def prepare_rave_dataset(directory: Path, manifest: dict, size: int = 512) -> LoopDataset:
    import urllib.parse

    clips = []
    for entry in manifest["clips"]:
        ds = import_video(
            directory / entry["file"],
            entry["name"],
            size,
            manifest["frames"],
            manifest["fps"],
            0,
            manifest["duration_seconds"],
            manifest["source_page"],
            manifest["license"],
            normalize_timestamps=True,
        )
        ds.sources[0].update(
            author=manifest["author"],
            source_file=entry["file"],
            download_url=manifest["download_root"] + urllib.parse.quote(entry["file"]),
            license_url=manifest["license_url"],
            description=entry["description"],
        )
        clips.append(ds)
    return merge_datasets(clips)
