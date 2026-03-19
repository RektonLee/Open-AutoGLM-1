"""Screenshot utilities for capturing HarmonyOS device screen."""

import base64
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

from PIL import Image
from phone_agent.hdc.connection import _run_hdc_command


# Image format configuration
IMAGE_FORMAT = "jpeg"  # "jpeg" or "png"
IMAGE_QUALITY = 85     # 0-100 for JPEG and png,  if choose png, it will be mapped to 0-9 for PNG compression

@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False


def get_screenshot(device_id: str | None = None, timeout: int = 10, quality: int | None = None, image_format: str | None = None) -> Screenshot:
    """
    Capture a screenshot from the connected HarmonyOS device.

    Args:
        device_id: Optional HDC device ID for multi-device setups.
        timeout: Timeout in seconds for screenshot operations.
        quality: Image quality (0-100). If None, uses IMAGE_QUALITY global setting.
                 For JPEG: 0-100 (higher = better quality, larger file).
                 For PNG: mapped to compression level 0-9.
        image_format: Image format ("jpeg" or "png"). If None, uses IMAGE_FORMAT global setting.

    Returns:
        Screenshot object containing base64 data and dimensions.

    Note:
        If the screenshot fails (e.g., on sensitive screens like payment pages),
        a black fallback image is returned with is_sensitive=True.
    """
    # Use global settings if not specified
    if quality is None:
        quality = IMAGE_QUALITY
    if image_format is None:
        image_format = IMAGE_FORMAT

    temp_path = os.path.join(tempfile.gettempdir(), f"screenshot_{uuid.uuid4()}.png")
    hdc_prefix = _get_hdc_prefix(device_id)

    try:
        # Execute screenshot command
        # HarmonyOS HDC only supports JPEG format
        remote_path = "/data/local/tmp/tmp_screenshot.jpeg"

        # Try method 1: hdc shell screenshot (newer HarmonyOS versions)
        result = _run_hdc_command(
            hdc_prefix + ["shell", "screenshot", remote_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for screenshot failure (sensitive screen)
        output = result.stdout + result.stderr
        if "fail" in output.lower() or "error" in output.lower() or "not found" in output.lower():
            # Try method 2: snapshot_display (older versions or different devices)
            result = _run_hdc_command(
                hdc_prefix + ["shell", "snapshot_display", "-f", remote_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            if "fail" in output.lower() or "error" in output.lower():
                return _create_fallback_screenshot(is_sensitive=True, quality=quality, image_format=image_format)

        # Pull screenshot to local temp path
        # Note: remote file is JPEG, but PIL can open it regardless of local extension
        _run_hdc_command(
            hdc_prefix + ["file", "recv", remote_path, temp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not os.path.exists(temp_path):
            return _create_fallback_screenshot(is_sensitive=False, quality=quality, image_format=image_format)

        # Read JPEG image and encode with specified format
        # PIL automatically detects the image format from file content
        img = Image.open(temp_path)
        width, height = img.size

        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if img.mode in ("RGBA", "LA", "P"):
            # Create a white background
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = rgb_img

        buffered = BytesIO()
        fmt = image_format.lower()
        if fmt in ["jpg", "jpeg"]:
            # JPEG format with quality compression
            img.save(buffered, format="JPEG", quality=int(quality))
        else:
            # PNG format: map quality (0-100) to compression level (0-9)
            comp_level = int(max(0, min(9, round((100 - int(quality)) / 10))))
            img.save(buffered, format="PNG", compress_level=comp_level)

        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Cleanup
        os.remove(temp_path)

        return Screenshot(
            base64_data=base64_data, width=width, height=height, is_sensitive=False
        )

    except Exception as e:
        print(f"Screenshot error: {e}")
        return _create_fallback_screenshot(is_sensitive=False, quality=quality, image_format=image_format)


def _get_hdc_prefix(device_id: str | None) -> list:
    """Get HDC command prefix with optional device specifier."""
    if device_id:
        return ["hdc", "-t", device_id]
    return ["hdc"]


def _create_fallback_screenshot(is_sensitive: bool, quality: int = 85, image_format: str = "jpeg") -> Screenshot:
    """
    Create a black fallback image when screenshot fails.

    Args:
        is_sensitive: Whether the screenshot failed due to sensitive content.
        quality: Image quality (0-100).
        image_format: Image format ("jpeg" or "png").
    """
    default_width, default_height = 1080, 2400

    black_img = Image.new("RGB", (default_width, default_height), color="black")
    buffered = BytesIO()

    fmt = image_format.lower()
    if fmt in ["jpg", "jpeg"]:
        black_img.save(buffered, format="JPEG", quality=int(quality))
    else:
        # PNG: map quality to compression level
        comp_level = int(max(0, min(9, round((100 - int(quality)) / 10))))
        black_img.save(buffered, format="PNG", compress_level=comp_level)

    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Screenshot(
        base64_data=base64_data,
        width=default_width,
        height=default_height,
        is_sensitive=is_sensitive,
    )
