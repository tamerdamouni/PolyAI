"""Tests for the image-processing tools the agent exposes.

The transform tools are thin wrappers around the img-proc MCP server; the MCP
call (`_call_mcp_tool`) is mocked so no server is needed. `get_detection_boxes`
is mocked at the httpx layer like `detect_objects`.
"""
import json
import os

os.environ.setdefault("MODEL", "openai.gpt-oss-20b-1:0")
os.environ.setdefault("AWS_REGION", "us-east-1")

from unittest.mock import MagicMock, patch

import app as app_module

KEY = "chat/img/original/image.jpg"
EDITED = "chat/img/original/edited-abc.png"


def test_blur_tool_calls_mcp_with_defaults_and_returns_key():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        result = app_module.blur.invoke({"image_key": KEY})
    assert result == EDITED
    mock_mcp.assert_called_once_with("blur", {"image_key": KEY, "radius": 2.0, "box": None})


def test_blur_tool_forwards_box_and_radius():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        app_module.blur.invoke({"image_key": KEY, "radius": 5.0, "box": [1, 2, 3, 4]})
    mock_mcp.assert_called_once_with(
        "blur", {"image_key": KEY, "radius": 5.0, "box": [1, 2, 3, 4]}
    )


def test_rotate_tool_calls_mcp():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        result = app_module.rotate.invoke({"image_key": KEY, "angle": 90})
    assert result == EDITED
    mock_mcp.assert_called_once_with("rotate", {"image_key": KEY, "angle": 90, "box": None})


def test_rotate_tool_forwards_box_for_object_specific_rotation():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        app_module.rotate.invoke({"image_key": KEY, "angle": 90, "box": [1, 2, 3, 4]})
    mock_mcp.assert_called_once_with(
        "rotate", {"image_key": KEY, "angle": 90, "box": [1, 2, 3, 4]}
    )


def test_flip_tool_forwards_box():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        app_module.flip.invoke({"image_key": KEY, "mode": "horizontal", "box": [1, 2, 3, 4]})
    mock_mcp.assert_called_once_with(
        "flip", {"image_key": KEY, "mode": "horizontal", "box": [1, 2, 3, 4]}
    )


def test_resize_tool_forwards_box():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        app_module.resize.invoke({"image_key": KEY, "width": 10, "height": 20, "box": [1, 2, 3, 4]})
    mock_mcp.assert_called_once_with(
        "resize", {"image_key": KEY, "width": 10, "height": 20, "box": [1, 2, 3, 4]}
    )


def test_crop_tool_requires_box():
    with patch.object(app_module, "_call_mcp_tool", return_value=EDITED) as mock_mcp:
        app_module.crop.invoke({"image_key": KEY, "box": [0, 0, 5, 5]})
    mock_mcp.assert_called_once_with("crop", {"image_key": KEY, "box": [0, 0, 5, 5]})


def test_all_transform_tools_registered():
    for name in ("rotate", "flip", "blur", "resize", "crop", "add_noise"):
        assert name in app_module.TOOLS
        assert name in app_module.TRANSFORM_TOOL_NAMES


def test_get_detection_boxes_parses_boxes_and_indexes():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "uid": "uid-1",
        "detection_objects": [
            {"label": "dog", "score": 0.9, "box": "[10.0, 20.0, 30.0, 40.0]"},
            {"label": "dog", "score": 0.8, "box": "[100.0, 20.0, 130.0, 40.0]"},
            {"label": "cat", "score": 0.7, "box": "[50.0, 60.0, 70.0, 80.0]"},
        ],
    }
    fake_response.raise_for_status.return_value = None
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.return_value = fake_response

    with patch.object(app_module.httpx, "Client", return_value=fake_client):
        result = app_module.get_detection_boxes.invoke({"prediction_id": "uid-1"})

    payload = json.loads(result)
    detections = payload["detections"]
    assert len(detections) == 3
    assert detections[0] == {"index": 0, "label": "dog", "score": 0.9, "box": [10.0, 20.0, 30.0, 40.0]}
    assert detections[1]["box"] == [100.0, 20.0, 130.0, 40.0]
    assert detections[2]["label"] == "cat"
