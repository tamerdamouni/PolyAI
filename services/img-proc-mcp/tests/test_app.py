import io

import pytest
from PIL import Image

import app


class FakeS3:
    """In-memory stand-in for the boto3 S3 client used by app.py."""

    def __init__(self):
        self.store = {}

    def download_fileobj(self, bucket, key, fileobj):
        fileobj.write(self.store[key])

    def upload_fileobj(self, fileobj, bucket, key):
        self.store[key] = fileobj.read()


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(app, "s3", fake)
    monkeypatch.setattr(app, "BUCKET", "test-bucket")
    return fake


def put_image(s3, key, size=(20, 20), color=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    s3.store[key] = buf.getvalue()


def open_stored(s3, key):
    return Image.open(io.BytesIO(s3.store[key]))


ORIGINAL = "chat/pred/original/image.jpg"


def test_blur_writes_new_object_and_returns_its_key(s3):
    put_image(s3, ORIGINAL)
    out_key = app.blur(ORIGINAL, radius=2.0)
    assert out_key != ORIGINAL
    assert out_key in s3.store


def test_output_key_is_colocated_with_input(s3):
    put_image(s3, ORIGINAL)
    out_key = app.blur(ORIGINAL)
    assert out_key.startswith("chat/pred/original/")
    assert out_key.endswith(".png")


def test_output_is_a_valid_image(s3):
    put_image(s3, ORIGINAL)
    out_key = app.blur(ORIGINAL)
    open_stored(s3, out_key).verify()


def test_resize_changes_stored_dimensions(s3):
    put_image(s3, ORIGINAL, size=(20, 20))
    out_key = app.resize(ORIGINAL, 8, 5)
    assert open_stored(s3, out_key).size == (8, 5)


def test_crop_returns_region_dimensions(s3):
    put_image(s3, ORIGINAL, size=(30, 30))
    out_key = app.crop(ORIGINAL, [5, 5, 15, 20])
    assert open_stored(s3, out_key).size == (10, 15)


def test_rotate_90_swaps_dimensions(s3):
    put_image(s3, ORIGINAL, size=(20, 10))
    out_key = app.rotate(ORIGINAL, 90)
    assert open_stored(s3, out_key).size == (10, 20)


def test_add_noise_region_leaves_corner_untouched(s3):
    put_image(s3, ORIGINAL, size=(40, 40), color=(128, 128, 128))
    out_key = app.add_noise(ORIGINAL, amount=0.9, box=[10, 10, 30, 30])
    out = open_stored(s3, out_key).convert("RGB")
    assert out.getpixel((0, 0)) == (128, 128, 128)


def test_rotate_region_preserves_full_image_dimensions(s3):
    put_image(s3, ORIGINAL, size=(40, 30))
    out_key = app.rotate(ORIGINAL, 90, box=[10, 5, 30, 25])
    assert open_stored(s3, out_key).size == (40, 30)


def test_flip_region_preserves_full_image_dimensions(s3):
    put_image(s3, ORIGINAL, size=(40, 30))
    out_key = app.flip(ORIGINAL, "horizontal", box=[10, 5, 30, 25])
    assert open_stored(s3, out_key).size == (40, 30)


def test_resize_region_preserves_full_image_dimensions(s3):
    put_image(s3, ORIGINAL, size=(40, 40))
    out_key = app.resize(ORIGINAL, 4, 4, box=[10, 10, 30, 30])
    assert open_stored(s3, out_key).size == (40, 40)


def test_flip_invalid_mode_raises(s3):
    put_image(s3, ORIGINAL)
    with pytest.raises(ValueError):
        app.flip(ORIGINAL, "diagonal")


def test_output_keys_are_unique(s3):
    put_image(s3, ORIGINAL)
    first = app.blur(ORIGINAL)
    second = app.blur(ORIGINAL)
    assert first != second


def test_all_six_tools_are_registered():
    registered = {fn.__name__ for fn in app.TOOL_FUNCTIONS}
    assert registered == {"rotate", "flip", "blur", "resize", "crop", "add_noise"}
