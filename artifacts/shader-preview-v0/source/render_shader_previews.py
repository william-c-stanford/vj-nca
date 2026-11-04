"""Render original shader studies and retain source/control provenance for training."""

import argparse
import html
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vj_nca.shader_data import ShaderRenderer, shader_source, validate_preset

ROOT = Path(__file__).resolve().parents[1]


def display_bytes(frame):
    return (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)


def gallery(output: Path, manifest: dict):
    cards = []
    for clip in manifest["clips"]:
        name = html.escape(clip["name"])
        stage = (
            "FIRST TRAINING STUDY"
            if clip["name"] == manifest.get("first_training_target")
            else "LATER CURRICULUM STUDY"
        )
        cards.append(
            f'<article><p class="eyebrow">{stage}</p><h2>{name.upper()}</h2>'
            "<video controls muted loop playsinline "
            f'poster="{name}.png" src="{name}.mp4"></video>'
            f"<p>{html.escape(clip['description'])}</p>"
            f'<a href="source/{name}.frag">GLSL source</a> · '
            f'<a href="{name}.json">Controls & provenance</a></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Kinetic line studies · shader targets</title><style>"
        "*{box-sizing:border-box}body{margin:0;padding:5vw;background:#10110f;color:#e1e5db;"
        "font-family:system-ui,sans-serif}header{max-width:850px;margin-bottom:56px}"
        "h1{font-size:clamp(36px,6vw,76px);font-weight:500;letter-spacing:-.05em;line-height:1}"
        ".eyebrow{font-size:12px;letter-spacing:.15em;color:#b1ce73}p{line-height:1.6;color:#a9ada3}"
        "main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:28px}"
        "video{display:block;width:100%;aspect-ratio:1;background:#000}h2{font-size:12px;"
        "font-weight:500;letter-spacing:.16em}a{color:#c7dc96;font-size:13px}"
        "footer{margin-top:70px;font-size:13px;color:#999} @media(max-width:850px){"
        "main{grid-template-columns:1fr}article{max-width:512px}} </style>"
        '<header><p class="eyebrow">VJ–NCA / ORIGINAL SHADER TARGETS / STUDY 01</p>'
        "<h1>Kinetic line studies</h1><p>Light, repetition and slow deformation. "
        f"{manifest['size']} × {manifest['size']} · {manifest['fps']} fps · "
        f"{manifest['duration_seconds']}-second loops. These are source shaders; "
        "NCA reconstruction is the next experiment.</p></header><main>"
        + "".join(cards)
        + "</main><footer>Original shaders: MIT. Rendered studies: CC0-1.0. "
        "No artist footage, external textures or pretrained model is used. "
        '<a href="manifest.json">Render manifest</a></footer></html>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/shader-preview-v0.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/shader-preview-v0")
    parser.add_argument("--stills-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output, size, fps = args.out, manifest["size"], manifest["fps"]
    frame_count = round(fps * manifest["duration_seconds"])
    if frame_count < 4 or fps <= 0:
        raise ValueError("Need a positive FPS and at least four frames")
    names = [c["name"] for c in manifest["clips"]]
    if len(set(names)) != len(names) or any(
        not n or n in {".", ".."} or Path(n).name != n for n in names
    ):
        raise ValueError("Clip names must be unique safe filenames")
    for clip in manifest["clips"]:
        validate_preset(clip, manifest["controls"])
    (output / "source").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, output / "source/presets.json")
    shutil.copyfile(__file__, output / "source/render_shader_previews.py")
    shutil.copyfile(
        ROOT / "scripts/probe_shader_learnability.py",
        output / "source/probe_shader_learnability.py",
    )
    shutil.copyfile(ROOT / "src/vj_nca/shader_data.py", output / "source/shader_data.py")
    shutil.copyfile(ROOT / "shaders/fullscreen.vert", output / "source/fullscreen.vert")
    started, records, panels = time.monotonic(), [], []
    with ShaderRenderer(ROOT / "shaders", size) as renderer:
        for clip in manifest["clips"]:
            name = clip["name"]
            (output / "source" / f"{name}.frag").write_text(
                shader_source(ROOT / "shaders", clip["shader"])
            )
            record = renderer.provenance(clip, manifest["controls"])
            zero, one = [renderer.render(clip, p) for p in (0, 1)]
            epsilon = 0.001
            before = renderer.render(clip, 1 - epsilon)
            after = renderer.render(clip, epsilon)
            # Centered derivatives at 0 and 1 use unreduced phase on both sides.
            derivative0 = (after - renderer.render(clip, -epsilon)) / (2 * epsilon)
            derivative1 = (renderer.render(clip, 1 + epsilon) - before) / (2 * epsilon)
            record["seam_max_error"] = float(np.abs(one - zero).max())
            record["seam_derivative_rmse"] = float(np.mean((derivative1 - derivative0) ** 2) ** 0.5)
            record["seam_derivative_relative_error"] = record["seam_derivative_rmse"] / max(
                float(np.mean(derivative0**2) ** 0.5), 1e-12
            )
            if (
                not np.isfinite(zero).all()
                or not np.isfinite(one).all()
                or not np.isfinite(derivative0).all()
                or not np.isfinite(derivative1).all()
                or record["seam_max_error"] > 0.001
                or record["seam_derivative_relative_error"] > 0.01
            ):
                raise ValueError(f"Non-finite output or discontinuous loop/velocity: {name}")
            samples = [renderer.render(clip, p) for p in (0, 0.25, 0.5, 0.75)]
            for index, sample in enumerate(samples):
                if not np.isfinite(sample).all() or sample.min() < -1e-5 or sample.max() > 1.00001:
                    raise ValueError(f"Invalid shader output: {name}")
                if index == 0:
                    Image.fromarray(display_bytes(sample)).save(output / f"{name}.png")
            panels.append([Image.fromarray(display_bytes(p)).resize((256, 256)) for p in samples])
            if not args.stills_only:
                command = [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{size}x{size}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "16",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output / f"{name}.mp4"),
                ]
                sums = np.zeros_like(zero, dtype=np.float64)
                square_sum = adjacent = mean_luma_min = 0.0
                mean_luma_max = 0.0
                previous, first, last = None, None, None
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    for frame in range(frame_count):
                        image = renderer.render(clip, frame / frame_count)
                        if not np.isfinite(image).all():
                            raise ValueError(f"Non-finite frame in {name}")
                        process.stdin.write(display_bytes(image).tobytes())
                        sums += image
                        square_sum += float(np.mean(image.astype(np.float64) ** 2))
                        luma = float((image @ np.array([0.2126, 0.7152, 0.0722])).mean())
                        mean_luma_min = min(mean_luma_min, luma) if frame else luma
                        mean_luma_max = max(mean_luma_max, luma)
                        if previous is not None:
                            adjacent += float(np.mean((image - previous) ** 2))
                        else:
                            first = image
                        previous, last = image, image
                    process.stdin.close()
                    error = process.stderr.read().decode()
                    if process.wait(timeout=60):
                        raise RuntimeError(error)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    process.stderr.close()
                seam = float(np.mean((first - last) ** 2))
                record.update(
                    frames=frame_count,
                    static_mean_mse=float(
                        square_sum / frame_count - np.mean((sums / frame_count) ** 2)
                    ),
                    adjacent_mse=adjacent / (frame_count - 1),
                    seam_adjacent_ratio=seam / max(adjacent / (frame_count - 1), 1e-12),
                    mean_luminance_range=[mean_luma_min, mean_luma_max],
                )
            records.append(record)
            (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    sheet = Image.new("RGB", (1024, len(panels) * 290), (15, 16, 14))
    draw = ImageDraw.Draw(sheet)
    for row, (name, samples) in enumerate(zip(names, panels, strict=True)):
        draw.text((12, row * 290 + 10), name.upper(), fill=(207, 224, 171))
        for col, sample in enumerate(samples):
            sheet.paste(sample, (col * 256, row * 290 + 34))
    sheet.save(output / "contact.png")
    result = {
        **manifest,
        "clips": records,
        "frames": frame_count,
        "render_seconds": time.monotonic() - started,
        "preview_only": args.stills_only,
        "nca_trained": False,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.stills_only:
        gallery(output, manifest)


if __name__ == "__main__":
    main()
