"""Image-processing MCP server.

Tools operate on S3 keys, never on image bytes: each reads an image from the
bucket, applies a transform, writes the result to a new key colocated with the
input, and returns that new key. Only text (keys, params) crosses the wire, so
the agent's LLM never sees image data.
"""
import io
import os
import posixpath
import uuid
from typing import Optional, Sequence

import boto3
from fastmcp import FastMCP
from PIL import Image

import transforms

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("AWS_S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

mcp = FastMCP("img-proc")

Box = Sequence[float]


def _load_image(key: str) -> Image.Image:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_image(img: Image.Image, key: str) -> None:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.upload_fileobj(buf, BUCKET, key)


def _output_key(input_key: str) -> str:
    """A unique key in the same S3 'folder' as the input."""
    folder = posixpath.dirname(input_key)
    name = f"edited-{uuid.uuid4().hex}.png"
    return posixpath.join(folder, name) if folder else name


def _process(input_key: str, transform) -> str:
    img = _load_image(input_key)
    out = transform(img)
    out_key = _output_key(input_key)
    _save_image(out, out_key)
    return out_key


def rotate(image_key: str, angle: float, box: Optional[Box] = None) -> str:
    """Rotate the image counter-clockwise by `angle` degrees, or only the region
    `box` [x1,y1,x2,y2] if given (the rotated region is refit into the box).

    Returns the new S3 key.
    """
    return _process(image_key, lambda img: transforms.rotate(img, angle, box=box))


def flip(image_key: str, mode: str, box: Optional[Box] = None) -> str:
    """Flip the image 'horizontal' or 'vertical', or only the region `box`
    [x1,y1,x2,y2] if given. Returns the new S3 key."""
    return _process(image_key, lambda img: transforms.flip(img, mode, box=box))


def blur(image_key: str, radius: float = 2.0, box: Optional[Box] = None) -> str:
    """Gaussian-blur the whole image, or only the region `box` [x1,y1,x2,y2] if given.

    Returns the new S3 key.
    """
    return _process(image_key, lambda img: transforms.blur(img, radius=radius, box=box))


def resize(image_key: str, width: int, height: int, box: Optional[Box] = None) -> str:
    """Resize the image to `width` x `height`, or only the region `box`
    [x1,y1,x2,y2] if given (the region's content is scaled then refit into the
    box, a resample effect on that object). Returns the new S3 key."""
    return _process(image_key, lambda img: transforms.resize(img, width, height, box=box))


def crop(image_key: str, box: Box) -> str:
    """Crop the region `box` [x1,y1,x2,y2] and return just that region as a new S3 key."""
    return _process(image_key, lambda img: transforms.crop(img, box))


def add_noise(image_key: str, amount: float = 0.05, box: Optional[Box] = None) -> str:
    """Add salt-and-pepper noise to the whole image, or only the region `box` if given.

    `amount` is the fraction of pixels affected. Returns the new S3 key.
    """
    return _process(image_key, lambda img: transforms.add_noise(img, amount=amount, box=box))


# Register every tool with the MCP server. Keeping the plain functions above lets
# the tools be unit-tested directly with a mocked S3 client.
TOOL_FUNCTIONS = [rotate, flip, blur, resize, crop, add_noise]
for _fn in TOOL_FUNCTIONS:
    mcp.tool()(_fn)


if __name__ == "__main__":
    mcp.run(transport="http", port=9000)
