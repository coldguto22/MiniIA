"""Pré-processamento e OCR com Tesseract, sem exigir hardware adicional."""

from __future__ import annotations

from typing import Any


def preprocess(image: Any, *, scale: int = 2, sharpness: float = 1.5) -> Any:
    """Amplia, reforça bordas e converte uma imagem Pillow para tons de cinza."""
    from PIL import ImageEnhance

    width, height = image.size
    enlarged = image.resize((width * scale, height * scale), resample=3)
    return ImageEnhance.Sharpness(enlarged).enhance(sharpness).convert("L")


def extract_text(image: Any, *, language: str = "por", config: str = "--psm 6",
                 preprocess_image: bool = True) -> str:
    """Extrai texto da imagem; Tesseract é carregado somente quando necessário."""
    import pytesseract

    source = preprocess(image) if preprocess_image else image
    return pytesseract.image_to_string(source, lang=language, config=config)


def capture_and_extract_text(*, language: str = "por") -> str:
    """Atalho compatível com a captura de tela antiga."""
    from dante.perception.screen import capture_screen

    return extract_text(capture_screen(), language=language)
