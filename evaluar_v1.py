"""Evalúa el modelo ORIGINAL (v1: ResNet50 + modelo_embeddings_A.joblib) en el MISMO test que v2.

Así la comparación antes/después es justa: mismas fotos, mismo ruido (misma semilla).
Ojo: v1 se entrenó con 300 fotos al azar por platillo de Food-101, que pueden incluir fotos del
test oficial, así que su resultado en Food-101 es, si acaso, optimista.

Salida: resultados/metricas_v1.json
Uso:  uv run python evaluar_v1.py
"""
import csv
import json
import os
import zlib

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from catalogo import EXTRA, REFUERZOS, SABORES
from comun import DATOS, abrir
from entrenar_clip import MEXICANAS, metricas
from ruido import ensuciar

TRANSF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Fotos(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, v = self.items[i]
        img = abrir(DATOS / ruta)
        if v > 0:
            img = ensuciar(img, zlib.crc32(ruta.encode()) * 16 + v)
        return TRANSF(img)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(DATOS / "indice.csv", encoding="utf-8") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    items = [(r["ruta"], v) for r in test for v in (0, 1)]
    if os.environ.get("LIMITE"):   # solo para probar rápido
        items = items[::len(items) // int(os.environ["LIMITE"]) | 1]

    datos = joblib.load("modelo_embeddings_A.joblib")
    clf, M, clases_v1 = datos["modelo"], datos["M"], list(datos["clases"])
    red = models.resnet50(weights="IMAGENET1K_V2")
    red.fc = torch.nn.Identity()
    red.eval().to(device)

    X = []
    with torch.inference_mode():
        for lote in DataLoader(Fotos(items), batch_size=64, num_workers=3):
            X.append(red(lote.to(device)).float().cpu().numpy())
    P = clf.predict_proba(np.vstack(X))

    clase = np.array([r for r, _ in items])
    meta = {r["ruta"]: r for r in test}
    clase = np.array([meta[r]["clase"] for r, _ in items])
    v = np.array([v for _, v in items])
    fuente = np.array([meta[r]["fuente"] for r, _ in items])
    Y = np.array([[int(meta[r][s]) for s in SABORES] for r, _ in items])
    # y = índice de platillo en v1 (-1 si v1 no conoce ese platillo)
    y = np.array([clases_v1.index(c) if c in clases_v1 else -1 for c in clase])
    mex = np.array([c in MEXICANAS or c in REFUERZOS for c in clase]) & (fuente == "web")

    grupos = {
        "todo_limpio": v == 0, "todo_ruido": v > 0,
        "food101_limpio": (v == 0) & (fuente == "food101"), "food101_ruido": (v > 0) & (fuente == "food101"),
        "mexicana_limpio": (v == 0) & mex, "mexicana_ruido": (v > 0) & mex,
        "web_limpio": (v == 0) & (fuente == "web"), "web_ruido": (v > 0) & (fuente == "web"),
    }
    res = {g: metricas(P[m], y[m], M, Y[m]) for g, m in grupos.items() if m.any()}
    res["nota"] = "platillo_top1 en grupos con platillos nuevos cuenta como error: v1 no los conoce"
    with open("resultados/metricas_v1.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for g in grupos:
        if g not in res:
            continue
        r = res[g]
        print(f"v1 {g:16s} platillo={r['platillo_top1']:.3f} F1macro={r['f1_macro']:.3f} "
              f"exacta={r['coincidencia_exacta']:.3f}")


if __name__ == "__main__":
    main()
