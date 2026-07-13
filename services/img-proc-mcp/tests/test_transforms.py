import pytest
from PIL import Image

from transforms import add_noise, blur, crop, flip, resize, rotate


def solid(width, height, color):
    return Image.new("RGB", (width, height), color)


def half_split():
    """A 4x2 image: left half red, right half blue."""
    img = Image.new("RGB", (4, 2), (255, 0, 0))
    for x in range(2, 4):
        for y in range(2):
            img.putpixel((x, y), (0, 0, 255))
    return img


def test_rotate_90_swaps_dimensions():
    img = solid(4, 2, (10, 20, 30))
    out = rotate(img, 90)
    assert out.size == (2, 4)


def test_rotate_returns_new_image():
    img = solid(4, 4, (0, 0, 0))
    out = rotate(img, 45)
    assert out is not img


def test_flip_horizontal_swaps_left_and_right():
    out = flip(half_split(), "horizontal")
    # Left column was red, after horizontal flip it should be blue.
    assert out.getpixel((0, 0)) == (0, 0, 255)
    assert out.getpixel((3, 0)) == (255, 0, 0)


def test_flip_vertical_swaps_top_and_bottom():
    img = Image.new("RGB", (2, 4), (255, 0, 0))
    for x in range(2):
        for y in range(2, 4):
            img.putpixel((x, y), (0, 0, 255))
    out = flip(img, "vertical")
    assert out.getpixel((0, 0)) == (0, 0, 255)
    assert out.getpixel((0, 3)) == (255, 0, 0)


def test_flip_invalid_mode_raises():
    with pytest.raises(ValueError):
        flip(solid(2, 2, (0, 0, 0)), "diagonal")


def test_resize_sets_exact_dimensions():
    out = resize(solid(4, 4, (0, 0, 0)), 10, 7)
    assert out.size == (10, 7)


def test_crop_returns_region_dimensions():
    out = crop(solid(10, 10, (0, 0, 0)), [2, 3, 8, 9])
    assert out.size == (6, 6)


def test_blur_changes_edge_pixels():
    out = blur(half_split(), radius=2.0)
    # The sharp red/blue boundary should smear, so a boundary pixel changes.
    assert out.getpixel((1, 0)) != (255, 0, 0)


def test_blur_region_leaves_outside_untouched():
    img = solid(20, 20, (100, 100, 100))
    # add a hard feature inside the region so blur has something to smear
    for x in range(12, 16):
        img.putpixel((x, 10), (0, 0, 0))
    out = blur(img, radius=3.0, box=[10, 5, 18, 15])
    # A pixel well outside the box is unchanged.
    assert out.getpixel((2, 2)) == (100, 100, 100)


def test_add_noise_changes_some_pixels():
    img = solid(50, 50, (128, 128, 128))
    out = add_noise(img, amount=0.2, seed=1)
    changed = sum(
        1
        for x in range(50)
        for y in range(50)
        if out.getpixel((x, y)) != (128, 128, 128)
    )
    assert changed > 0


def test_add_noise_region_leaves_outside_untouched():
    img = solid(50, 50, (128, 128, 128))
    out = add_noise(img, amount=0.5, box=[20, 20, 40, 40], seed=1)
    assert out.getpixel((0, 0)) == (128, 128, 128)
    assert out.getpixel((49, 49)) == (128, 128, 128)


def test_add_noise_is_deterministic_with_seed():
    img = solid(30, 30, (128, 128, 128))
    a = add_noise(img, amount=0.3, seed=42)
    b = add_noise(img, amount=0.3, seed=42)
    assert a.tobytes() == b.tobytes()
