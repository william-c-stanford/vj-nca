import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from vj_nca.media import save_video


class MediaTests(unittest.TestCase):
    def test_gif_preserves_average_playback_rate_with_centisecond_timestamps(self):
        video = torch.linspace(-1, 1, 17).reshape(17, 1, 1, 1).expand(17, 3, 4, 4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rate.gif"
            save_video(path, video, fps=16)
            with Image.open(path) as image:
                durations = []
                for index in range(image.n_frames):
                    image.seek(index)
                    durations.append(image.info["duration"])
            self.assertLessEqual(abs(sum(durations) - 17 / 16 * 1000), 10)


if __name__ == "__main__":
    unittest.main()
