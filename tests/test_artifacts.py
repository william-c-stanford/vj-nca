import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from vj_nca.artifacts import archive_directory, extract_archive


class ArtifactTests(unittest.TestCase):
    def test_epoch_dated_modal_source_survives_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.mkdir()
            file = source / "model.py"
            file.write_text("epoch-dated mounted source\n")
            os.utime(file, (0, 0))
            with self.assertRaisesRegex(ValueError, "timestamps before 1980"):
                with zipfile.ZipFile(io.BytesIO(), "w") as old:
                    old.write(file, "model.py")
            payload = archive_directory(source, prefix="source")
            extract_archive(payload, root / "out")
            self.assertEqual((root / "out/source/model.py").read_text(), file.read_text())

    def test_artifact_cannot_write_outside_output_directory(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../escape.txt", "bad")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                extract_archive(stream.getvalue(), root / "out")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
