"""Entrena el clasificador de platillos sobre los vectores de SigLIP2 y evalúa los sabores.

Modelo:  vector SigLIP2 (768) -> capa lineal -> softmax sobre 146 platillos
         P(sabor) = P(platillo) @ M            (M = tabla platillo -> sabor, catalogo.py)
         Se calibra una temperatura T para que la "certeza" signifique lo que dice.

Evaluación (en fotos de test que el modelo nunca vio, ni sus copias con ruido):
  - accuracy de platillo top-1 / top-3, limpio y con ruido
  - F1 por sabor, F1 macro, coincidencia exacta de las 5 etiquetas, exactitud por etiqueta
  - lo mismo solo en comida mexicana (platillos nuevos + refuerzos)
  - línea base "todo es salado" y error de calibración (ECE)

Salida: modelo_sabores_v2.npz y resultados/metricas.json (+ CSVs para el reporte).
Uso:  uv run python entrenar_clip.py
"""
import csv
import glob
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score

from catalogo import EXTRA, REFUERZOS, SABORES, tabla, vector_sabores
from comun import ARCH, ARCHIVO_MODELO, DATOS, PESOS

torch.manual_seed(0)
np.random.seed(0)
MEXICANAS = set(EXTRA) - {"cafe_negro", "chocolate_amargo", "toronja", "limones", "ensalada_de_arugula"}


def cargar():
    meta = {}
    with open(DATOS / "indice.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["ruta"]] = r
    X, rutas, v = [], [], []
    for archivo in sorted(glob.glob(str(DATOS / "emb" / "*.npz"))):
        z = np.load(archivo)
        ok = z["ok"] & np.array([r in meta for r in z["ruta"]])
        X.append(z["X"][ok]), rutas.append(z["ruta"][ok]), v.append(z["v"][ok])
    X = np.vstack(X).astype(np.float32)
    rutas, v = np.concatenate(rutas), np.concatenate(v)
    return X, rutas, v, meta


def entrenar_lineal(X, y, w_clase, n_clases, Xv=None, yv=None, epocas=60, lr=2e-3, wd=1e-4, device="cpu"):
    lin = torch.nn.Linear(X.shape[1], n_clases).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    pesos = torch.tensor(w_clase, dtype=torch.float32, device=device)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)
    escala = 30.0  # los vectores vienen normalizados (norma 1): se escalan para que el softmax no sea plano
    historia = []
    for ep in range(epocas):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            perdida(lin(Xt[b] * escala), yt[b]).backward()
            opt.step()
        sched.step()
        if Xv is not None and (ep + 1) % 5 == 0:
            with torch.no_grad():
                acc = (lin(torch.tensor(Xv, device=device) * escala).argmax(1).cpu().numpy() == yv).mean()
            historia.append((ep + 1, float(acc)))
    W = lin.weight.detach().cpu().numpy() * escala
    b = lin.bias.detach().cpu().numpy()
    return W, b, historia


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calibrar_T(logits, y):
    mejor, T_mejor = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(5), 200)):
        p = softmax(logits / T)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < mejor:
            mejor, T_mejor = nll, float(T)
    return T_mejor


def ece(conf, acierto, bins=10):
    total, e = len(conf), 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.sum() / total * abs(conf[m].mean() - acierto[m].mean())
    return float(e)


def metricas(P_plato, y, M, Y_sabor):
    top = np.argsort(-P_plato, 1)
    P_sabor = np.clip(P_plato @ M, 0, 1)
    pred = (P_sabor > 0.5).astype(int)
    f1 = f1_score(Y_sabor, pred, average=None, zero_division=0)
    siempre_salado = np.zeros_like(Y_sabor)
    siempre_salado[:, SABORES.index("salado")] = 1
    return {
        "n": int(len(y)),
        "platillo_top1": float((top[:, 0] == y).mean()),
        "platillo_top3": float((top[:, :3] == y[:, None]).any(1).mean()),
        "f1_por_sabor": dict(zip(SABORES, map(float, f1))),
        "f1_macro": float(f1.mean()),
        "coincidencia_exacta": float((pred == Y_sabor).all(1).mean()),
        "exactitud_por_etiqueta": float((pred == Y_sabor).mean()),
        "base_siempre_salado_f1_macro": float(f1_score(Y_sabor, siempre_salado, average=None,
                                                         zero_division=0).mean()),
        "base_siempre_salado_exacta": float((siempre_salado == Y_sabor).all(1).mean()),
        "ece_platillo": ece(P_plato.max(1), (top[:, 0] == y).astype(float)),
        "certeza_media": float(P_plato.max(1).mean()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tab = tabla()
    clases = sorted(tab)
    cidx = {c: i for i, c in enumerate(clases)}
    M = np.array([vector_sabores(tab[c][0]) for c in clases], dtype=np.float32)

    X, rutas, v, meta = cargar()
    clase = np.array([meta[r]["clase"] for r in rutas])
    y = np.array([cidx[c] for c in clase])
    split = np.array([meta[r]["split"] for r in rutas])
    fuente = np.array([meta[r]["fuente"] for r in rutas])
    Y_sabor = M[y].astype(int)
    print(f"{len(X)} vectores | train {np.sum(split == 'train')} | test {np.sum(split == 'test')}")

    # Validación: 10 % de las FOTOS de train (con todas sus copias) para calibrar T y elegir épocas
    fotos_train = np.unique(rutas[split == "train"])
    rng = np.random.default_rng(0)
    val_fotos = set(rng.choice(fotos_train, size=len(fotos_train) // 10, replace=False))
    es_val = np.array([r in val_fotos for r in rutas]) & (split == "train")
    es_tr = (split == "train") & ~es_val

    conteo = np.bincount(y[es_tr], minlength=len(clases)).astype(float)
    w_clase = (conteo.max() / np.maximum(conteo, 1)) ** 0.5   # compensa clases con menos fotos

    # 1) Elegir el número de épocas con validación
    _, _, historia = entrenar_lineal(X[es_tr], y[es_tr], w_clase, len(clases), X[es_val], y[es_val],
                                     epocas=40, device=device)
    mejor_ep = max(historia, key=lambda h: h[1])[0]
    print("accuracy en validación por época:", historia, "-> mejor:", mejor_ep)
    # 2) Modelo final (solo train, sin validación) y 3) temperatura calibrada en validación.
    #    Se usa el MISMO modelo que se calibró, para que el % de certeza sea honesto.
    W, b, _ = entrenar_lineal(X[es_tr], y[es_tr], w_clase, len(clases), epocas=mejor_ep, device=device)
    T = calibrar_T(X[es_val] @ W.T + b, y[es_val])
    print("temperatura calibrada:", round(T, 3))
    tr = es_tr

    te = split == "test"
    P = softmax((X @ W.T + b) / T)
    mex = np.array([c in MEXICANAS or c in REFUERZOS for c in clase]) & (fuente == "web")
    grupos = {
        "todo_limpio": te & (v == 0), "todo_ruido": te & (v > 0),
        "food101_limpio": te & (v == 0) & (fuente == "food101"), "food101_ruido": te & (v > 0) & (fuente == "food101"),
        "mexicana_limpio": te & (v == 0) & mex, "mexicana_ruido": te & (v > 0) & mex,
        "web_limpio": te & (v == 0) & (fuente == "web"), "web_ruido": te & (v > 0) & (fuente == "web"),
    }
    res = {g: metricas(P[m], y[m], M, Y_sabor[m]) for g, m in grupos.items() if m.any()}
    res["temperatura"] = T
    res["historia_validacion"] = historia
    res["epocas"] = int(mejor_ep)
    res["fotos_validacion"] = len(val_fotos)
    res["extractor"] = f"{ARCH} ({PESOS})"
    res["n_clases"] = len(clases)
    res["fotos_por_split"] = {s: int(len(np.unique(rutas[split == s]))) for s in ("train", "test")}
    res["vectores_train"] = int(tr.sum())

    # Accuracy por clase y confusiones más comunes (test con ruido + limpio)
    por_clase = defaultdict(lambda: [0, 0, 0, 0])
    pred = P.argmax(1)
    confusiones = defaultdict(int)
    for i in np.where(te)[0]:
        k = 0 if v[i] == 0 else 2
        por_clase[clase[i]][k] += int(pred[i] == y[i])
        por_clase[clase[i]][k + 1] += 1
        if pred[i] != y[i]:
            confusiones[(clase[i], clases[pred[i]])] += 1
    with open("resultados/accuracy_por_platillo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "nombre", "sabores", "fuente", "acc_limpio", "acc_ruido", "n_test"])
        for c in clases:
            a = por_clase[c]
            w.writerow([c, tab[c][1], tab[c][0], "nuevo" if c in EXTRA else "food101",
                        round(a[0] / max(a[1], 1), 4), round(a[2] / max(a[3], 1), 4), a[1]])
    res["confusiones_top"] = [[tab[a][1], tab[b][1], n] for (a, b), n in
                              sorted(confusiones.items(), key=lambda t: -t[1])[:15]]

    # Curva de confiabilidad (para gráfica): certeza vs acierto en test con ruido
    m = grupos["todo_ruido"]
    conf, ac = P[m].max(1), (pred[m] == y[m])
    res["confiabilidad_ruido"] = [[float(lo), float(conf[(conf > lo) & (conf <= lo + .1)].mean()) if ((conf > lo) & (conf <= lo + .1)).any() else None,
                                   float(ac[(conf > lo) & (conf <= lo + .1)].mean()) if ((conf > lo) & (conf <= lo + .1)).any() else None,
                                   int(((conf > lo) & (conf <= lo + .1)).sum())] for lo in np.arange(0, 1, .1)]

    with open("resultados/metricas.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(ARCHIVO_MODELO, W=W.astype(np.float32), b=b.astype(np.float32), T=np.float32(T),
                        clases=np.array(clases), nombres=np.array([tab[c][1] for c in clases]),
                        letras=np.array([tab[c][0] for c in clases]), M=M, sabores=np.array(SABORES),
                        arch=np.array(ARCH), pesos=np.array(PESOS))
    for g in grupos:
        if g in res:
            r = res[g]
            print(f"{g:16s} platillo={r['platillo_top1']:.3f} top3={r['platillo_top3']:.3f} "
                  f"F1macro={r['f1_macro']:.3f} exacta={r['coincidencia_exacta']:.3f} n={r['n']}")
    print(f"Guardado {ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()
