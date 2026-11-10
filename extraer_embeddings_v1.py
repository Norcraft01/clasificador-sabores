"""[v1] Convierte cada foto de etiquetas.csv en un vector de 2048 números usando ResNet50
preentrenado (sin entrenarlo). Usa la GPU si hay, si no, CPU. Se corre UNA sola vez.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

RUTA = "food-101/images"
SALIDA = "embeddings_resnet50.npy"
BATCH = 128
WORKERS = 6   # hilos que decodifican los JPG (con CPU de 6 núcleos va bien)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

transformar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Fotos(Dataset):
    def __init__(self, rutas):
        self.rutas = rutas

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        return transformar(Image.open(f"{RUTA}/{self.rutas[i]}").convert("RGB"))


df = pd.read_csv("etiquetas.csv")
loader = DataLoader(Fotos(df["imagen"].tolist()), batch_size=BATCH,
                    num_workers=WORKERS, shuffle=False)

modelo = models.resnet50(weights="IMAGENET1K_V2")
modelo.fc = torch.nn.Identity()   # quitamos la capa final: ahora devuelve 2048 números
modelo.eval().to(device)

vectores = []
with torch.inference_mode():
    for i, lote in enumerate(loader):
        lote = lote.to(device)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                v = modelo(lote)
        else:
            v = modelo(lote)
        vectores.append(v.float().cpu().numpy())
        print(f"\rLote {i + 1}/{len(loader)}", end="", flush=True)

X = np.vstack(vectores)
np.save(SALIDA, X)
print(f"\nListo: {X.shape} guardado en {SALIDA}")
