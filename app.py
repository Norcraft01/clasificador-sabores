"""Interfaz web (Gradio) del clasificador de sabores v2.

Uso:  uv run python app.py      -> abre http://127.0.0.1:7860
"""
import html
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import ImageOps

from comun import ARCHIVO_MODELO, cargar_vision, embeber
from ruido import ensuciar

UMBRAL_SABOR = 0.5      # un sabor se marca si su probabilidad pasa de 50 %
UMBRAL_DUDA = 0.40      # si el platillo más probable tiene menos de 40 %, se avisa

ICONO = {"acido": "🍋", "amargo": "☕", "dulce": "🍬", "picante": "🌶️", "salado": "🧂"}
NOMBRE = {"acido": "Ácido", "amargo": "Amargo", "dulce": "Dulce", "picante": "Picante", "salado": "Salado"}
COLOR = {"acido": "#c9d100", "amargo": "#6b4226", "dulce": "#ff5fa2", "picante": "#e63212", "salado": "#3a86c8"}

z = np.load(ARCHIVO_MODELO)
W, b, T, M = z["W"], z["b"], float(z["T"]), z["M"]
NOMBRES, LETRAS, SABORES = z["nombres"], z["letras"], list(z["sabores"])
modelo, pre, device = cargar_vision()


def analizar(img, con_ruido, semilla):
    if img is None:
        return None, "<div class='vacio'>Sube una foto de comida 👆</div>"
    # fotos de celular: respetar la orientación EXIF y quitar transparencia (PNG)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if con_ruido:
        img = ensuciar(img, int(semilla))
    v = embeber(modelo, pre(img).unsqueeze(0), device)[0]
    logits = (v @ W.T + b) / T
    p = np.exp(logits - logits.max())
    p /= p.sum()
    p_sabor = np.clip(p @ M, 0, 1)
    top = np.argsort(-p)[:3]
    return img, tarjeta(p, top, p_sabor)


def tarjeta(p, top, p_sabor):
    certeza = p[top[0]]
    detectados = [s for s, q in zip(SABORES, p_sabor) if q > UMBRAL_SABOR]
    chips = "".join(f"<span class='chip' style='background:{COLOR[s]}'>{ICONO[s]} {NOMBRE[s]}</span>"
                    for s in detectados) or "<span class='chip gris'>Ningún sabor pasó del 50 %</span>"
    barras = "".join(
        f"<div class='fila'><span class='et'>{ICONO[s]} {NOMBRE[s]}</span>"
        f"<div class='barra'><div style='width:{q * 100:.0f}%;background:{COLOR[s]}'></div></div>"
        f"<span class='num'>{q * 100:.0f}%</span></div>" for s, q in zip(SABORES, p_sabor))
    otros = "".join(f"<li>{html.escape(str(NOMBRES[i]))} <b>{p[i] * 100:.1f}%</b></li>" for i in top[1:])
    aviso = ("<div class='aviso'>🤔 No estoy muy seguro: puede ser un platillo que no conozco "
             "o la foto está muy dañada. Revisa las otras opciones.</div>") if certeza < UMBRAL_DUDA else ""
    return f"""
    <div class='tarjeta'>
      <div class='plato'>{html.escape(str(NOMBRES[top[0]]))}</div>
      <div class='certeza'>Certeza del platillo: <b>{certeza * 100:.1f}%</b></div>
      <div class='chips'>{chips}</div>
      {aviso}
      <h4>Probabilidad de cada sabor</h4>{barras}
      <h4>También podría ser</h4><ol>{otros}</ol>
    </div>"""


CSS = """
.gradio-container {background: #fff8e7 !important; font-family: 'Trebuchet MS', sans-serif;}
#titulo {text-align:center; padding: 6px 0 0 0;}
#titulo h1 {font-size: 2.3em; margin: 0; color:#d6006f; text-shadow: 3px 3px 0 #ffd23f;}
#titulo p {color:#555; margin: 4px 0 0 0;}
.papel {height: 26px; margin-bottom: 6px;
  background: repeating-linear-gradient(90deg,#d6006f 0 40px,#ff8c00 40px 80px,#ffd23f 80px 120px,
                                         #00a86b 120px 160px,#00a3e0 160px 200px,#8e44ad 200px 240px);
  -webkit-mask: radial-gradient(circle 10px at 20px 26px, transparent 98%, #000) 0 0/40px 26px;
          mask: radial-gradient(circle 10px at 20px 26px, transparent 98%, #000) 0 0/40px 26px;}
.tarjeta {background:white; border:4px dashed #d6006f; border-radius:18px; padding:18px 22px;
          box-shadow: 6px 6px 0 #ffd23f;}
.plato {font-size: 2em; font-weight: bold; color:#1d3557;}
.certeza {font-size: 1.1em; color:#444; margin-bottom: 8px;}
.chip {display:inline-block; color:white; font-weight:bold; padding:6px 14px; margin:4px 6px 4px 0;
       border-radius: 999px; font-size: 1.05em; box-shadow: 2px 2px 0 #0003;}
.chip.gris {background:#999;}
.fila {display:flex; align-items:center; gap:8px; margin:5px 0;}
.et {width: 110px;} .num {width: 44px; text-align:right; font-weight:bold;}
.barra {flex:1; height:16px; background:#eee; border-radius:8px; overflow:hidden;}
.barra div {height:100%; border-radius:8px;}
.aviso {background:#fff3cd; border-left:6px solid #ff8c00; padding:8px 12px; margin:10px 0; border-radius:6px;}
.vacio {text-align:center; font-size:1.3em; color:#888; padding:60px 0;}
h4 {margin: 14px 0 4px 0; color:#d6006f;}
footer {display:none !important;}
"""

ejemplos = sorted(str(p) for p in Path("ejemplos").glob("*.jpg"))

with gr.Blocks(css=CSS, title="¿A qué sabe?", theme=gr.themes.Soft(primary_hue="pink")) as demo:
    gr.HTML("<div class='papel'></div><div id='titulo'><h1>🌮 ¿A qué sabe? 🌶️</h1>"
            "<p>Sube una foto de comida y te digo qué platillo es y si es ácido, amargo, dulce, picante "
            "y/o salado.</p></div>")
    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Image(type="pil", label="Tu foto", height=330)
            with gr.Row():
                con_ruido = gr.Checkbox(label="Ensuciar la foto (prueba de ruido)")
                semilla = gr.Slider(0, 99, value=7, step=1, label="Tipo de ruido")
            boton = gr.Button("¡Adivina el sabor!", variant="primary", size="lg")
            if ejemplos:
                gr.Examples(ejemplos, inputs=entrada, label="Fotos de ejemplo")
        with gr.Column(scale=1):
            vista = gr.Image(label="Lo que ve el modelo", height=230, interactive=False)
            salida = gr.HTML("<div class='vacio'>Sube una foto de comida 👆</div>")
    gr.HTML("<p style='text-align:center;color:#999'>Conoce 146 platillos (Food-101 + comida mexicana). "
            "SigLIP2 + regresión lineal · Machine Learning, ESCOM-IPN</p>")
    for ev in (boton.click, entrada.upload):
        ev(analizar, [entrada, con_ruido, semilla], [vista, salida])
    con_ruido.change(analizar, [entrada, con_ruido, semilla], [vista, salida])
    semilla.release(analizar, [entrada, con_ruido, semilla], [vista, salida])

if __name__ == "__main__":
    demo.launch(inbrowser=os.environ.get("SIN_NAVEGADOR") is None)
