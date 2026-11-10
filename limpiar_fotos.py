"""Limpia las fotos descargadas de internet (datos/extra/<clase>/) antes de entrenar.

Para cada foto:
  1. Válida: se puede abrir y mide al menos 150x150 px.
  2. Es comida: SigLIP2 compara la foto con textos de comida/bebida/postre contra textos negativos
     (menú, logotipo, persona, dibujo, receta escrita, collage). Si gana un negativo, se descarta.
  3. Es el platillo: SigLIP2 ordena los 146 platillos por parecido (zero-shot); la clase buscada
     debe quedar dentro de los primeros TOP_K (8).
  4. No es repetida: si su vector se parece > UMBRAL_DUP (coseno) a una foto ya aceptada, se descarta.

Salida: datos/extra_validas.csv y resultados/limpieza.csv (conteos por clase).
Uso:  uv run python limpiar_fotos.py
"""
import csv
from pathlib import Path

import numpy as np
import torch

from catalogo import tabla
from comun import DATOS, abrir, cargar_extractor

TOP_K = 8
# Textos especiales para nombres ambiguos ("barbacoa" en inglés es BBQ)
TEXTO_CLASE = {"barbacoa": "barbacoa mexicana de borrego o res, en tacos o en plato",
               "carnitas": "carnitas de cerdo mexicanas, en tacos o en plato"}
UMBRAL_DUP = 0.95
POSITIVOS = ["una foto de un platillo de comida", "una foto de una bebida en una taza o un vaso",
             "una foto de fruta, dulces o postres"]
NEGATIVOS = ["una foto de un menú o de texto", "un logotipo", "una foto de una persona posando",
             "un dibujo o una ilustración", "una receta escrita", "un collage de varias fotos",
             "una foto de una tienda o un restaurante por fuera"]


def main():
    import argparse
    import open_clip
    from comun import ARCH

    ap = argparse.ArgumentParser()
    ap.add_argument("--clases", nargs="*", help="solo estas clases (para probar)")
    ap.add_argument("--sufijo", default="", help="sufijo para los archivos de salida (para probar)")
    args = ap.parse_args()

    modelo, pre, device = cargar_extractor()
    tok = open_clip.get_tokenizer(ARCH)
    tab = tabla()
    clases = sorted(tab)

    @torch.inference_mode()
    def textos(frases):
        t = modelo.encode_text(tok(frases).to(device))
        return torch.nn.functional.normalize(t.float(), dim=-1)

    T_clases = textos([f"una foto de {TEXTO_CLASE.get(c, tab[c][1])}, comida" for c in clases])
    T_neg = textos(POSITIVOS + NEGATIVOS)

    validas, resumen = [], []
    for carpeta in sorted(p for p in (DATOS / "extra").iterdir() if p.is_dir()):
        clase = carpeta.name
        if args.clases and clase not in args.clases:
            continue
        idx = clases.index(clase)
        cont = {"clase": clase, "descargadas": 0, "invalidas": 0, "no_comida": 0,
                "otro_platillo": 0, "duplicadas": 0, "validas": 0}
        aceptadas = []
        archivos = sorted(carpeta.iterdir())
        cont["descargadas"] = len(archivos)
        for i in range(0, len(archivos), 32):
            imgs, rutas = [], []
            for a in archivos[i:i + 32]:
                try:
                    img = abrir(a)
                    if min(img.size) < 150:
                        raise ValueError("chica")
                    imgs.append(pre(img))
                    rutas.append(a)
                except Exception:
                    cont["invalidas"] += 1
            if not imgs:
                continue
            with torch.inference_mode():
                V = torch.nn.functional.normalize(
                    modelo.encode_image(torch.stack(imgs).to(device)).float(), dim=-1)
            es_comida = (V @ T_neg.T).argmax(1) < len(POSITIVOS)
            rango = ((V @ T_clases.T) > (V @ T_clases[idx:idx + 1].T)).sum(1)  # 0 = primera
            for v, ruta, com, rg in zip(V.cpu().numpy(), rutas, es_comida.tolist(), rango.tolist()):
                if not com:
                    cont["no_comida"] += 1
                elif rg >= TOP_K:
                    cont["otro_platillo"] += 1
                elif aceptadas and max(float(v @ w) for w in aceptadas) > UMBRAL_DUP:
                    cont["duplicadas"] += 1
                else:
                    aceptadas.append(v)
                    validas.append({"ruta": ruta.relative_to(DATOS).as_posix(), "clase": clase, "rango": rg})
        cont["validas"] = len(aceptadas)
        resumen.append(cont)
        print(cont, flush=True)

    with open(DATOS / f"extra_validas{args.sufijo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "clase", "rango"])
        w.writeheader()
        w.writerows(validas)
    Path("resultados").mkdir(exist_ok=True)
    with open(f"resultados/limpieza{args.sufijo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)
    print(f"Total válidas: {len(validas)}")


if __name__ == "__main__":
    main()
