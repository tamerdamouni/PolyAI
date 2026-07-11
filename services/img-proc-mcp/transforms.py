"""Pure image transforms: PIL Image in, PIL Image out. No S3, no MCP.

Region-capable transforms (blur, add_noise) take an optional bounding box
[x1, y1, x2, y2]. When given, the transform is applied only to that region and
pasted back into a copy of the full image.
"""
import random
from typing import Callable, Optional, Sequence

from PIL import Image, ImageFilter

Box = Sequence[float]


def _apply_to_region(img: Image.Image, box: Box, fn: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """Apply fn to the region `box` only, and paste the result back in place.

    If the transform changed the region's dimensions (e.g. a rotation), the result
    is squeezed back to the box's exact size so it always fits the hole it came
    from. The image's overall size and everything outside the box are untouched.
    """
    left, upper, right, lower = (int(round(c)) for c in box)
    region = img.crop((left, upper, right, lower))
    transformed = fn(region)
    if transformed.size != region.size:
        transformed = transformed.resize(region.size)
    result = img.copy()
    result.paste(transformed, (left, upper))
    return result


def rotate(img: Image.Image, angle: float, box: Optional[Box] = None) -> Image.Image:
    """Rotate the whole image counter-clockwise by angle degrees, expanding to fit.

    With a box, only that region is rotated and then refit into the box, so the
    object looks rotated (and squeezed) in place.
    """
    def _rotate(target: Image.Image) -> Image.Image:
        return target.rotate(angle, expand=True)

    if box is None:
        return _rotate(img)
    return _apply_to_region(img, box, _rotate)


def flip(img: Image.Image, mode: str, box: Optional[Box] = None) -> Image.Image:
    """Flip the whole image horizontally or vertically, or only the region `box`."""
    def _flip(target: Image.Image) -> Image.Image:
        if mode == "horizontal":
            return target.transpose(Image.FLIP_LEFT_RIGHT)
        if mode == "vertical":
            return target.transpose(Image.FLIP_TOP_BOTTOM)
        raise ValueError(f"mode must be 'horizontal' or 'vertical', got {mode!r}")

    if box is None:
        return _flip(img)
    return _apply_to_region(img, box, _flip)


def resize(img: Image.Image, width: int, height: int, box: Optional[Box] = None) -> Image.Image:
    """Resize the whole image to width x height.

    With a box, the region's content is scaled to width x height and then refit
    into the box — a resample effect on that object, leaving the image size intact.
    """
    def _resize(target: Image.Image) -> Image.Image:
        return target.resize((width, height))

    if box is None:
        return _resize(img)
    return _apply_to_region(img, box, _resize)


def crop(img: Image.Image, box: Box) -> Image.Image:
    """Crop the region [x1, y1, x2, y2] and return just that region."""
    left, upper, right, lower = (int(round(c)) for c in box)
    return img.crop((left, upper, right, lower))


def blur(img: Image.Image, radius: float = 2.0, box: Optional[Box] = None) -> Image.Image:
    """Apply a Gaussian blur to the whole image, or to a region if box is given."""
    def _blur(target: Image.Image) -> Image.Image:
        return target.filter(ImageFilter.GaussianBlur(radius))

    if box is None:
        return _blur(img.copy())
    return _apply_to_region(img, box, _blur)


def add_noise(
    img: Image.Image,
    amount: float = 0.05,
    box: Optional[Box] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """Add salt-and-pepper noise to the whole image, or to a region if box is given.

    `amount` is the fraction of pixels turned to pure black or white.
    """
    def _noise(target: Image.Image) -> Image.Image:
        rng = random.Random(seed)
        out = target.convert("RGB")
        pixels = out.load()
        w, h = out.size
        for y in range(h):
            for x in range(w):
                if rng.random() < amount:
                    value = 0 if rng.random() < 0.5 else 255
                    pixels[x, y] = (value, value, value)
        return out

    if box is None:
        return _noise(img.copy())
    return _apply_to_region(img, box, _noise)
