"""Captura de tela isolada da etapa de OCR."""

from __future__ import annotations

from typing import Any


def capture_screen() -> Any:
    """Captura o monitor primário e retorna uma imagem Pillow RGB."""
    import mss
    from PIL import Image

    with mss.MSS() as capture:
        shot = capture.grab(capture.monitors[1])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
