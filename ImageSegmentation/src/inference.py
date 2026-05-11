from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import io

from PIL import Image
from rembg import remove

OutputMode = Literal["transparent", "white"]


@dataclass
class ModelConfig:
    ref_size: int = 512
    device: str | None = None


class BackgroundRemover:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def remove(self, image: Image.Image, output_mode: OutputMode = "transparent") -> Image.Image:
        rgba = image.convert("RGBA")
        output = remove(rgba)

        if isinstance(output, bytes):
            output_image = Image.open(io.BytesIO(output)).convert("RGBA")
        else:
            output_image = output.convert("RGBA")

        if output_mode == "transparent":
            return output_image

        white_bg = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_bg, output_image).convert("RGB")
