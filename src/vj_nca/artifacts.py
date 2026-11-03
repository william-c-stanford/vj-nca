import io
import zipfile
from pathlib import Path


def archive_directory(root: Path, prefix: str = "", paths: list[Path] | None = None) -> bytes:
    """Create a portable ZIP, including sources mounted with Unix-epoch timestamps."""
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False
    ) as archive:
        for path in sorted(paths if paths is not None else root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(prefix) / path.relative_to(root))
    return stream.getvalue()


def extract_archive(payload: bytes, output: Path):
    """Extract a trusted worker's result only within the chosen output directory."""
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if not (output / member.filename).resolve().is_relative_to(output.resolve()):
                raise ValueError("Unsafe artifact path")
        archive.extractall(output)
