"""Descarga fotos de los platillos nuevos (y refuerzos mexicanos) con búsqueda de imágenes de Bing.

Las fotos quedan en datos/extra/<clase>/ y NO se suben al repo. Después hay que correr
`limpiar_fotos.py`, que quita duplicados y fotos que no son del platillo.

Uso:  uv run python descargar_fotos.py [--por-clase 600] [--solo clase1 clase2 ...]
"""
import argparse
import logging
import math
from pathlib import Path

from icrawler.builtin import BingImageCrawler

from catalogo import EXTRA, REFUERZOS

RAIZ = Path("datos/extra")


def descargar(clase, busquedas, total):
    carpeta = RAIZ / clase
    carpeta.mkdir(parents=True, exist_ok=True)
    por_busqueda = math.ceil(total / len(busquedas))
    for q in busquedas:
        crawler = BingImageCrawler(storage={"root_dir": str(carpeta)},
                                   downloader_threads=8, log_level=logging.ERROR)
        crawler.crawl(keyword=q, max_num=por_busqueda, min_size=(200, 200),
                      file_idx_offset="auto")
    print(f"{clase}: {len(list(carpeta.iterdir()))} archivos", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--por-clase", type=int, default=600, help="fotos a pedir por platillo nuevo")
    p.add_argument("--refuerzo", type=int, default=400, help="fotos a pedir por clase reforzada")
    p.add_argument("--solo", nargs="*")
    a = p.parse_args()

    trabajos = [(c, v[2], a.por_clase) for c, v in EXTRA.items()]
    trabajos += [(c, qs, a.refuerzo) for c, qs in REFUERZOS.items()]
    for clase, busquedas, total in trabajos:
        if a.solo and clase not in a.solo:
            continue
        descargar(clase, busquedas, total)


if __name__ == "__main__":
    main()
