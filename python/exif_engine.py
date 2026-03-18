"""Extract EXIF metadata from images using Pillow."""
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def _get_exif_data(image: Image.Image) -> dict:
    raw = image._getexif()  # type: ignore[attr-defined]
    if not raw:
        return {}
    return {TAGS.get(tag, tag): value for tag, value in raw.items()}


def _dms_to_decimal(dms, ref: str) -> Optional[float]:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except Exception:
        return None


def _extract_gps(exif: dict) -> tuple[Optional[float], Optional[float]]:
    gps_info = exif.get("GPSInfo")
    if not gps_info:
        return None, None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", ""))
    lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", ""))
    return lat, lon


def _parse_date(date_str: str) -> Optional[datetime]:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            pass
    return None


def extract_metadata(path: str) -> dict:
    """
    Returns a dict with keys:
      width, height, format, date_taken, gps_lat, gps_lon, camera_model
    """
    result: dict = {
        "width": None,
        "height": None,
        "format": None,
        "date_taken": None,
        "gps_lat": None,
        "gps_lon": None,
        "camera_model": None,
    }
    try:
        with Image.open(path) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["format"] = img.format or Path(path).suffix.lstrip(".").upper()

            exif = {}
            if hasattr(img, "_getexif") and img._getexif() is not None:
                exif = _get_exif_data(img)

            date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
            if date_str:
                result["date_taken"] = _parse_date(str(date_str))

            result["gps_lat"], result["gps_lon"] = _extract_gps(exif)
            result["camera_model"] = exif.get("Model")
    except Exception as exc:
        # Non-fatal: return partial data
        pass
    return result
