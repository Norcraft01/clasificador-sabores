"""Piezas compartidas: carga del extractor CLIP, rutas y lectura de imágenes."""
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

DATOS = Path("datos")
ARCH, PESOS = "ViT-B-16-SigLIP2", "webli"     # extractor elegido (ver README: comparación)
ARCHIVO_MODELO = Path("modelo_sabores_v2.npz")


def dispositivo():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cargar_extractor(arch=ARCH, pesos=PESOS, device=None):
    """Devuelve (modelo_visual, preprocesado, device). En fp32: la GTX 16xx es lenta en fp16."""
    import open_clip
    device = device or dispositivo()
    modelo, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pesos)
    modelo = modelo.eval().to(device)
    return modelo, pre, device


PESOS_VISION = Path("pesos/siglip2_vision_fp16.safetensors")


class _Vision(torch.nn.Module):
    """Envoltura para que la red de timm tenga el mismo método encode_image que open_clip."""
    def __init__(self, red):
        super().__init__()
        self.red = red

    def encode_image(self, x):
        return self.red(x)


def cargar_vision(device=None):
    """Solo la parte visual de SigLIP2 (186 MB, sin internet) para la app y las predicciones.

    Da los mismos vectores que open_clip (coseno > 0.99999). Si no existe el archivo de pesos,
    usa open_clip y los descarga de Hugging Face.
    """
    if not PESOS_VISION.exists():
        return cargar_extractor(device=device)
    import timm
    from safetensors.torch import load_file
    from torchvision import transforms
    device = device or dispositivo()
    red = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0)
    red.load_state_dict({k: v.float() for k, v in load_file(str(PESOS_VISION)).items()})
    pre = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return _Vision(red).eval().to(device), pre, device


@torch.inference_mode()
def embeber(modelo, lote, device):
    """lote: tensor (n,3,H,W) ya preprocesado -> embeddings L2-normalizados (n, d) float32."""
    v = modelo.encode_image(lote.to(device))
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


def abrir(ruta):
    img = Image.open(ruta)
    img.draft("RGB", (640, 640))   # JPEG: decodifica a tamaño reducido (más rápido)
    return img.convert("RGB")


def guardar_npz(ruta, **arrays):
    np.savez_compressed(ruta, **arrays)
