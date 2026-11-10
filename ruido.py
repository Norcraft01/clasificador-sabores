"""Generador de ruido para fotos: simula fotos de mala calidad (como las que meterá el profe).

Cada foto ruidosa recibe un ajuste geométrico leve (recorte, giro, espejo) y de 1 a 3
degradaciones elegidas al azar entre:
  - ruido gaussiano (grano)          - ruido sal y pimienta (puntos blancos/negros)
  - desenfoque                       - compresión JPEG muy baja
  - baja resolución (pixelado)       - brillo / contraste / saturación extremos
  - ruido de color por canal

Se usa un generador con semilla para que el mismo número de foto siempre dé el mismo ruido
(así la evaluación es reproducible).
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _gaussiano(img, rng):
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, rng.uniform(10, 45), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _sal_pimienta(img, rng):
    a = np.asarray(img).copy()
    p = rng.uniform(0.01, 0.10)
    m = rng.random(a.shape[:2])
    a[m < p / 2] = 0
    a[m > 1 - p / 2] = 255
    return Image.fromarray(a)


def _desenfoque(img, rng):
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 3.5)))


def _jpeg(img, rng):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(rng.integers(5, 30)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _pixelado(img, rng):
    w, h = img.size
    f = rng.uniform(0.12, 0.35)
    peque = img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
    return peque.resize((w, h), Image.NEAREST if rng.random() < 0.5 else Image.BILINEAR)


def _iluminacion(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.4, 1.6))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.4, 1.6))
    return ImageEnhance.Color(img).enhance(rng.uniform(0.2, 1.8))


def _color(img, rng):
    a = np.asarray(img, dtype=np.float32)
    a = a * rng.uniform(0.7, 1.3, 3) + rng.normal(0, rng.uniform(5, 25), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


DEGRADACIONES = [_gaussiano, _sal_pimienta, _desenfoque, _jpeg, _pixelado, _iluminacion, _color]


def _geometria(img, rng):
    w, h = img.size
    f = rng.uniform(0.65, 1.0)
    cw, ch = int(w * f), int(h * f)
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-25, 25), resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    return img


def aplicar(img, tipo, nivel):
    """Ruido de UN tipo con intensidad fija (para la prueba controlada). nivel: 1 leve, 2 medio, 3 fuerte."""
    img = img.convert("RGB")
    img.thumbnail((512, 512))
    rng = np.random.default_rng(nivel)
    if tipo == "gaussiano":
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, [15, 35, 60][nivel - 1], (img.size[1], img.size[0], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if tipo == "sal_pimienta":
        a = np.asarray(img).copy()
        m = rng.random(a.shape[:2])
        p = [0.03, 0.08, 0.15][nivel - 1]
        a[m < p / 2], a[m > 1 - p / 2] = 0, 255
        return Image.fromarray(a)
    if tipo == "desenfoque":
        return img.filter(ImageFilter.GaussianBlur([1.5, 3, 5][nivel - 1]))
    if tipo == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=[20, 8, 3][nivel - 1])
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if tipo == "pixelado":
        w, h = img.size
        f = [0.35, 0.2, 0.1][nivel - 1]
        return img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR).resize((w, h), Image.NEAREST)
    if tipo == "combinado":   # grano + desenfoque + JPEG a la vez
        for t in ("desenfoque", "gaussiano", "jpeg"):
            img = aplicar(img, t, nivel)
        return img
    raise ValueError(tipo)


TIPOS_PRUEBA = ["gaussiano", "sal_pimienta", "desenfoque", "jpeg", "pixelado", "combinado"]


def ensuciar(img, semilla):
    """Devuelve una copia ruidosa de `img` (PIL RGB). Misma semilla -> mismo ruido."""
    rng = np.random.default_rng(semilla)
    img = img.convert("RGB")
    # Trabajar a <=512 px para que el ruido tenga la misma intensidad en fotos grandes y chicas
    img.thumbnail((512, 512))
    img = _geometria(img, rng)
    n = rng.integers(1, 4)
    for i in rng.choice(len(DEGRADACIONES), size=n, replace=False):
        img = DEGRADACIONES[i](img, rng)
    return img
