"""Convierte cada foto de datos/indice.csv en un vector con SigLIP2 (sin entrenar la red).

Por cada foto se guardan varias "variantes":
  v=0          la foto limpia
  v=1..N       copias con ruido distinto (ver ruido.py)
Train lleva --ruido-train copias ruidosas (más para fotos web, que son menos) y test lleva 1,
para poder medir el desempeño con y sin ruido por separado.

Es reanudable: guarda bloques en datos/emb/ y, si se interrumpe, continúa donde se quedó.

Uso:  uv run python extraer_embeddings.py [--fuente food101|web|todas]
"""
import argparse
import csv
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from comun import DATOS, abrir, cargar_extractor, embeber
from ruido import ensuciar

SALIDA = DATOS / "emb"
BLOQUE = 8192


class Variantes(Dataset):
    def __init__(self, items, pre):
        self.items, self.pre = items, pre

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, v = self.items[i]
        try:
            img = abrir(DATOS / ruta)
            if v > 0:
                img = ensuciar(img, zlib.crc32(ruta.encode()) * 16 + v)
            return self.pre(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fuente", default="todas")
    p.add_argument("--ruido-train", type=int, default=1, help="copias ruidosas por foto de train (Food-101)")
    p.add_argument("--ruido-train-web", type=int, default=2, help="copias ruidosas por foto web de train")
    p.add_argument("--workers", type=int, default=7)
    a = p.parse_args()

    with open(DATOS / "indice.csv", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if a.fuente in ("todas", r["fuente"])]

    items = []
    for r in filas:
        n = 1 if r["split"] == "test" else (a.ruido_train_web if r["fuente"] == "web" else a.ruido_train)
        items += [(r["ruta"], v) for v in range(n + 1)]
    print(f"{len(filas)} fotos -> {len(items)} variantes a extraer", flush=True)

    SALIDA.mkdir(parents=True, exist_ok=True)
    modelo, pre, device = cargar_extractor()
    etiqueta = a.fuente
    for b, ini in enumerate(range(0, len(items), BLOQUE)):
        destino = SALIDA / f"{etiqueta}_{b:04d}.npz"
        if destino.exists():
            continue
        parte = items[ini:ini + BLOQUE]
        loader = DataLoader(Variantes(parte, pre), batch_size=64, num_workers=a.workers,
                            persistent_workers=False, pin_memory=device == "cuda")
        X, ok = [], []
        for lote, bien in loader:
            X.append(embeber(modelo, lote, device).astype(np.float16))
            ok.append(bien.numpy())
        np.savez(destino, X=np.vstack(X), ok=np.concatenate(ok),
                 ruta=np.array([r for r, _ in parte]), v=np.array([v for _, v in parte], dtype=np.int8))
        print(f"bloque {b + 1}/{-(-len(items) // BLOQUE)} listo", flush=True)
    print("Extracción terminada.")


if __name__ == "__main__":
    main()
