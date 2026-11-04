"""Optional offscreen GLSL targets; no dependency on an NCA checkpoint."""

import hashlib
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np


def shader_source(directory: Path, filename: str) -> str:
    if Path(filename).name != filename or not filename.endswith(".frag"):
        raise ValueError("Shader filename must name a .frag file in the shader directory")
    return (
        "#version 330\n"
        + (directory / "common.glsl").read_text()
        + (directory / filename).read_text()
    )


def validate_preset(preset: dict, controls: dict) -> dict[str, float]:
    values = preset["parameters"]
    if set(values) != set(controls):
        raise ValueError("Preset must specify exactly the declared controls")
    result = {}
    for name, spec in controls.items():
        value = float(values[name])
        if not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"Control outside declared range: {name}")
        if spec["max"] <= spec["min"]:
            raise ValueError(f"Invalid control range: {name}")
        result[name] = (value - spec["min"]) / (spec["max"] - spec["min"])
    return result


class ShaderRenderer:
    """Native-pixel float output. Caller owns the context lifetime via `with`."""

    def __init__(self, directory: Path, size: int = 512):
        import moderngl

        if size < 16 or size % 2:
            raise ValueError("Use an even render size >=16")
        self.directory, self.size = directory, size
        self.context = moderngl.create_standalone_context(require=330)
        self.texture = self.context.texture((size, size), 4, dtype="f4")
        self.framebuffer = self.context.framebuffer(color_attachments=[self.texture])
        self.framebuffer.use()
        self.context.viewport = (0, 0, size, size)
        self.programs = {}

    def render(self, preset: dict, phase: float) -> np.ndarray:
        if not math.isfinite(phase):
            raise ValueError("Phase must be finite")
        filename = preset["shader"]
        if filename not in self.programs:
            program = self.context.program(
                vertex_shader=(self.directory / "fullscreen.vert").read_text(),
                fragment_shader=shader_source(self.directory, filename),
            )
            self.programs[filename] = (program, self.context.vertex_array(program, []))
        program, vao = self.programs[filename]
        program["u_resolution"].value = (self.size, self.size)
        # Do not reduce phase modulo one here: seam checks must exercise GLSL.
        program["u_phase"].value = phase
        for name, value in preset["parameters"].items():
            program[f"u_{name}"].value = value
        vao.render(vertices=3)
        pixels = self.framebuffer.read(components=3, dtype="f4", alignment=1)
        return np.frombuffer(pixels, np.float32).reshape(self.size, self.size, 3)[::-1].copy()

    def provenance(self, preset: dict, controls: dict) -> dict:
        source = shader_source(self.directory, preset["shader"])
        return {
            **preset,
            "normalized_controls": validate_preset(preset, controls),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "description_sha256": hashlib.sha256(preset["description"].encode()).hexdigest(),
            "vertex_sha256": hashlib.sha256(
                (self.directory / "fullscreen.vert").read_bytes()
            ).hexdigest(),
            "renderer": self.context.info["GL_RENDERER"],
            "opengl_version": self.context.info["GL_VERSION"],
            "moderngl_version": version("moderngl"),
            "glcontext_version": version("glcontext"),
            "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for program, vao in self.programs.values():
            vao.release()
            program.release()
        self.framebuffer.release()
        self.texture.release()
        self.context.release()
