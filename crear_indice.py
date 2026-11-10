"""Crea datos/indice.csv: una fila por foto con su ruta, platillo, split (train/test) y sabores.

- Food-101: se usan las 1,000 fotos de cada platillo con el split OFICIAL (750 train / 250 test).
- Platillos nuevos y refuerzos: las fotos que aprobó `limpiar_fotos.py` (datos/extra_validas.csv),
  separadas 80/20 al azar por clase con semilla fija.

Uso:  uv run python crear_indice.py
"""
import csv
import random

from catalogo import SABORES, tabla, vector_sabores
from comun import DATOS

F101 = DATOS / "food-101"


def main():
    tab = tabla()
    filas = []
    for split, archivo in (("train", "train.txt"), ("test", "test.txt")):
        for linea in (F101 / "meta" / archivo).read_text().split():
            clase = linea.split("/")[0]
            filas.append({"ruta": f"food-101/images/{linea}.jpg", "clase": clase, "split": split,
                          "fuente": "food101"})

    validas = DATOS / "extra_validas.csv"
    if validas.exists():
        por_clase = {}
        with open(validas, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                por_clase.setdefault(r["clase"], []).append(r["ruta"])
        rnd = random.Random(42)
        for clase, rutas in sorted(por_clase.items()):
            rutas.sort()
            rnd.shuffle(rutas)
            corte = int(len(rutas) * 0.8)
            for i, ruta in enumerate(rutas):
                filas.append({"ruta": ruta, "clase": clase, "split": "train" if i < corte else "test",
                              "fuente": "web"})
    else:
        print("AVISO: no existe datos/extra_validas.csv; el índice solo tendrá Food-101.")

    for r in filas:
        r.update(dict(zip(SABORES, vector_sabores(tab[r["clase"]][0]))))
    with open(DATOS / "indice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "clase", "split", "fuente", *SABORES])
        w.writeheader()
        w.writerows(filas)
    print(f"indice.csv: {len(filas)} fotos, {len({r['clase'] for r in filas})} platillos")


if __name__ == "__main__":
    main()
