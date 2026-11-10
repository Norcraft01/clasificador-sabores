"""Compara extractores CLIP en una muestra de Food-101 (rápido) para elegir el mejor.

Por clase: 30 fotos de entrenamiento y 15 de prueba (split oficial). Se mide la accuracy de
platillo con una regresión logística, en prueba limpia y en prueba con ruido.
"""
import random
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from comun import DATOS, abrir, cargar_extractor, embeber
from ruido import ensuciar

CANDIDATOS = [("ViT-B-16", "openai"), ("ViT-B-16-SigLIP2", "webli"), ("ViT-B-16", "datacomp_xl_s13b_b90k")]
F101 = DATOS / "food-101"


def muestra(archivo, n):
    por_clase = {}
    for linea in (F101 / "meta" / archivo).read_text().split():
        por_clase.setdefault(linea.split("/")[0], []).append(linea)
    random.seed(0)
    return [(r, c) for c, rs in sorted(por_clase.items()) for r in random.sample(rs, n)]


def extraer(modelo, pre, device, items, ruido):
    X = []
    for i in range(0, len(items), 32):
        imgs = []
        for j, (r, _) in enumerate(items[i:i + 32]):
            img = abrir(F101 / "images" / f"{r}.jpg")
            imgs.append(pre(ensuciar(img, i + j) if ruido else img))
        X.append(embeber(modelo, torch.stack(imgs), device))
    return np.vstack(X)


def main():
    tr, te = muestra("train.txt", 30), muestra("test.txt", 15)
    y_tr, y_te = [c for _, c in tr], [c for _, c in te]
    for arch, pesos in CANDIDATOS:
        modelo, pre, device = cargar_extractor(arch, pesos)
        Xtr, Xte = extraer(modelo, pre, device, tr, False), extraer(modelo, pre, device, te, False)
        Xte_r = extraer(modelo, pre, device, te, True)
        clf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, y_tr)
        print(f"{arch:20s} {pesos:24s} limpio={clf.score(Xte, y_te):.3f}  ruido={clf.score(Xte_r, y_te):.3f}",
              flush=True)
        del modelo
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
