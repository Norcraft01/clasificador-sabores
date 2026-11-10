"""Prueba "tipo profe": fotos NUEVAS (no usadas en entrenamiento) + ruido controlado.

Pasos:
  uv run python prueba_profe.py descargar   # baja ~20 fotos por platillo con búsquedas distintas
  uv run python prueba_profe.py evaluar     # evalúa v2 (y v1 para comparar) con 6 tipos x 3 niveles de ruido

Se descartan las fotos que no son comida (puestos, letreros; mismo filtro que limpiar_fotos.py) y
las que se parecen (coseno > 0.93) a alguna foto de entrenamiento, para que la prueba mida
generalización real y no memoria.
Salida: resultados/prueba_profe.json, resultados/prueba_profe_detalle.csv
"""
import csv
import glob
import os
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catalogo import SABORES, tabla, vector_sabores
from comun import ARCHIVO_MODELO, DATOS, abrir, cargar_extractor, embeber
from ruido import TIPOS_PRUEBA, aplicar

RAIZ = DATOS / "prueba_profe"
# platillo -> búsquedas (distintas a las de entrenamiento: contexto casero, celular, puesto, etc.)
PLATOS = {
    "tacos": ["tacos de puesto callejero foto celular", "orden de tacos con salsa casera"],
    "enchiladas": ["enchiladas caseras en plato", "enchiladas potosinas"],
    "chilaquiles": ["chilaquiles desayuno fonda", "chilaquiles con pollo y frijoles"],
    "pozole": ["plato de pozole casero con tostadas", "pozole fiestas patrias"],
    "tamales": ["tamales verdes rojos en vaporera", "tamal de rajas con queso"],
    "mole_poblano": ["mole con arroz plato casero", "pollo en mole rojo"],
    "elote": ["elote en vaso puesto callejero", "elote con queso y chile en palito"],
    "sopes": ["sopes caseros de chorizo", "sopes de pollo con lechuga"],
    "chiles_en_nogada": ["chile en nogada restaurante septiembre"],
    "birria": ["tacos de birria con consomé", "plato de birria tapatía"],
    "aguachile": ["aguachile negro de camarón", "aguachile en molcajete"],
    "pan_dulce": ["pan dulce mexicano panadería charola", "concha de vainilla pan"],
    "flan": ["rebanada de flan casero"],
    "guacamole": ["guacamole con totopos casero"],
    "chicken_quesadilla": ["quesadilla de queso oaxaca en comal"],
    "churros": ["churros con azúcar y canela recién hechos"],
    "nachos": ["nachos con queso amarillo cine"],
    "huevos_rancheros": ["huevos rancheros desayuno casero"],
    "pizza": ["rebanada de pizza de pepperoni"],
    "hamburger": ["hamburguesa casera con papas"],
    "sushi": ["rollos de sushi empanizado mexicano"],
    "ice_cream": ["helado de chocolate en cono"],
    "cafe_negro": ["taza de café negro en la mañana"],
    "ceviche": ["tostada de ceviche de pescado"],
    "chocolate_cake": ["rebanada de pastel de chocolate casero"],
}


def descargar():
    from icrawler.builtin import BingImageCrawler
    for clase, qs in PLATOS.items():
        carpeta = RAIZ / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for q in qs:
            BingImageCrawler(storage={"root_dir": str(carpeta)}, downloader_threads=6,
                             log_level=logging.CRITICAL).crawl(keyword=q, max_num=30 // len(qs) + 8,
                                                               min_size=(200, 200), file_idx_offset="auto")
        print(clase, len(list(carpeta.iterdir())), flush=True)


def evaluar(*solo):
    tab = tabla()
    z = np.load(ARCHIVO_MODELO)
    W, b, T, M = z["W"], z["b"], float(z["T"]), z["M"]
    clases = list(z["clases"])
    modelo, pre, device = cargar_extractor(str(z["arch"]), str(z["pesos"]))

    # Filtro "¿es comida?" (el mismo de limpiar_fotos.py): descarta puestos, letreros, personas...
    # NO se filtra por platillo, para no quedarse solo con las fotos fáciles.
    import open_clip
    from limpiar_fotos import NEGATIVOS, POSITIVOS
    with torch.inference_mode():
        T_neg = torch.nn.functional.normalize(modelo.encode_text(
            open_clip.get_tokenizer(str(z["arch"]))(POSITIVOS + NEGATIVOS).to(device)).float(), dim=-1)

    # Vectores de entrenamiento (limpios) para descartar fotos casi idénticas
    train = []
    with open(DATOS / "indice.csv", encoding="utf-8") as f:
        meta = {r["ruta"]: r["split"] for r in csv.DictReader(f)}
    for a in sorted(glob.glob(str(DATOS / "emb" / "*.npz"))):
        e = np.load(a)
        m = (e["v"] == 0) & np.array([meta.get(r) == "train" for r in e["ruta"]])
        train.append(e["X"][m])
    Xtr = torch.tensor(np.vstack(train).astype(np.float32), device=device)

    # v1 para comparar
    import joblib
    from torchvision import models, transforms
    v1 = joblib.load("modelo_embeddings_A.joblib")
    red = models.resnet50(weights="IMAGENET1K_V2")
    red.fc = torch.nn.Identity()
    red.eval().to(device)
    tf1 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    condiciones = [("limpia", 0)] + [(t, n) for t in TIPOS_PRUEBA for n in (1, 2, 3)]
    filas, descartadas, no_comida = [], 0, 0
    for clase in PLATOS:
        if solo and clase not in solo:
            continue
        Y = np.array(vector_sabores(tab[clase][0]))
        for ruta in sorted((RAIZ / clase).glob("*"))[:int(os.environ.get("PROFE_MAX", 999))]:
            try:
                img0 = abrir(ruta)
            except Exception:
                continue
            v0 = embeber(modelo, pre(img0).unsqueeze(0), device)
            v0t = torch.tensor(v0, device=device)
            if int((v0t @ T_neg.T).argmax()) >= len(POSITIVOS):
                no_comida += 1
                continue
            if float((v0t @ Xtr.T).max()) > 0.93:
                descartadas += 1
                continue
            imgs = [img0 if t == "limpia" else aplicar(img0, t, n) for t, n in condiciones]
            V = embeber(modelo, torch.stack([pre(i) for i in imgs]), device)
            P = np.exp((V @ W.T + b) / T - ((V @ W.T + b) / T).max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            with torch.inference_mode():
                F1 = red(torch.stack([tf1(i) for i in imgs]).to(device)).cpu().numpy()
            P1 = v1["modelo"].predict_proba(F1)
            for (t, n), p, p1 in zip(condiciones, P, P1):
                s2 = (np.clip(p @ M, 0, 1) > 0.5).astype(int)
                s1 = (np.clip(p1 @ v1["M"], 0, 1) > 0.5).astype(int)
                filas.append({"foto": ruta.as_posix(), "clase": clase, "ruido": t, "nivel": n,
                              "v2_platillo": clases[p.argmax()], "v2_certeza": round(float(p.max()), 4),
                              "v2_ok_platillo": int(clases[p.argmax()] == clase),
                              "v2_exacta": int((s2 == Y).all()), "v2_etiquetas_ok": float((s2 == Y).mean()),
                              "v1_exacta": int((s1 == Y).all()), "v1_etiquetas_ok": float((s1 == Y).mean()),
                              "v1_ok_platillo": int(str(v1["clases"][p1.argmax()]) == clase),
                              **{f"v2_{s}": int(x) for s, x in zip(SABORES, s2)},
                              **{f"real_{s}": int(x) for s, x in zip(SABORES, Y)}})
        print(clase, "ok", flush=True)

    with open("resultados/prueba_profe_detalle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    res = {"fotos": len({r["foto"] for r in filas}), "descartadas_por_parecido": descartadas, "descartadas_no_comida": no_comida,
           "platillos": len(PLATOS), "por_condicion": {}, "por_platillo": {}}
    grupos = defaultdict(list)
    for r in filas:
        grupos[f"{r['ruido']}_{r['nivel']}"].append(r)
    for g, rs in grupos.items():
        res["por_condicion"][g] = {k: float(np.mean([r[k] for r in rs])) for k in
                                   ("v2_ok_platillo", "v2_exacta", "v2_etiquetas_ok", "v2_certeza",
                                    "v1_ok_platillo", "v1_exacta", "v1_etiquetas_ok")}
    for clase in PLATOS:
        rs = [r for r in filas if r["clase"] == clase]
        if rs:
            res["por_platillo"][clase] = {"n_fotos": len({r["foto"] for r in rs}),
                                          "v2_exacta_limpia": float(np.mean([r["v2_exacta"] for r in rs if r["ruido"] == "limpia"])),
                                          "v2_exacta_ruido": float(np.mean([r["v2_exacta"] for r in rs if r["ruido"] != "limpia"])),
                                          "v1_exacta_ruido": float(np.mean([r["v1_exacta"] for r in rs if r["ruido"] != "limpia"]))}
    todas = [r for r in filas if r["ruido"] != "limpia"]
    res["global_ruido"] = {k: float(np.mean([r[k] for r in todas])) for k in
                           ("v2_ok_platillo", "v2_exacta", "v2_etiquetas_ok", "v1_ok_platillo", "v1_exacta", "v1_etiquetas_ok")}
    with open("resultados/prueba_profe.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res["global_ruido"], indent=1), res["fotos"], "fotos")


if __name__ == "__main__":
    {"descargar": descargar, "evaluar": evaluar}[sys.argv[1]](*sys.argv[2:])
