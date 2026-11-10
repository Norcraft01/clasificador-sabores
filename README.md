# Clasificador de sabores en comida — v2

> Foto de comida → platillo + sabores (ácido, amargo, dulce, picante y/o salado) con su porcentaje de certeza. Aguanta fotos con ruido y conoce comida mexicana.

Esta es la **versión 2** del proyecto del equipo (rama `v2-clip-comida-mexicana`). La rama `main` conserva la versión original sin cambios, por si hay que regresar a ella.

## ¿Qué cambió respecto a la v1?

| | v1 (main) | v2 (esta rama) |
|---|---|---|
| Extractor de características | ResNet50 (ImageNet) | **SigLIP2 ViT-B/16** (Google, 2025) |
| Platillos | 101 (Food-101) | **146** (Food-101 + 45 nuevos: 39 mexicanos y 6 de refuerzo para amargo y ácido) |
| Fotos | 30,300 (300 por platillo) | **113,976** (Food-101 completo + 12,976 de internet filtradas) |
| Ruido | No | **Entrena con una copia ruidosa de cada foto** (grano, sal y pimienta, desenfoque, JPEG, pixelado, iluminación) |
| Certeza | Probabilidad sin calibrar | **Calibrada** (temperatura) y aviso si el modelo duda |
| Tabla platillo → sabor | La del equipo | **La misma** para los 101 originales; nuevas filas para los 45 platillos agregados |
| Interfaz | Gradio básica | Gradio con tarjeta de resultados, barras por sabor y botón de "ensuciar la foto" |

Los resultados completos (antes contra después, con y sin ruido) están en `resultados/` y en el reporte (`docs/`).

## Cómo correrlo (Windows, lo más fácil)

1. Descomprime el zip.
2. Doble clic en **`INICIAR.bat`**.
   - La primera vez instala `uv` (gestor de Python) y las dependencias (~1 GB, tarda unos minutos).
   - Después solo abre la app en el navegador (`http://127.0.0.1:7860`).

## Cómo correrlo (cualquier sistema, a mano)

Requisito: [uv](https://docs.astral.sh/uv/) (`pip install uv` también sirve).

```bash
uv run --extra cpu python app.py      # sin tarjeta NVIDIA (suficiente para usar la app)
uv run --extra gpu python app.py      # con tarjeta NVIDIA (CUDA 12.6)
```

uv descarga Python 3.12 y todo lo necesario. En el zip los pesos vienen incluidos (`pesos/` y `modelo_sabores_v2.npz`), así que la app no descarga ningún modelo. En el repositorio `pesos/` no se sube (186 MB, más que el límite de GitHub): si falta, la app descarga SigLIP2 de Hugging Face la primera vez (~800 MB).

## Reentrenar desde cero (opcional, requiere GPU NVIDIA y ~10 GB libres)

Todos los comandos con `uv run --extra gpu python ...`:

| Paso | Script | Qué hace | Tiempo (GTX 1650) |
|---|---|---|---|
| 1 | `descargar_fotos.py` | Baja fotos de los 45 platillos nuevos y refuerzos mexicanos (Bing) | ~1.5 h |
| 2 | `limpiar_fotos.py` | Quita fotos rotas, que no son comida, de otro platillo o repetidas (SigLIP2 zero-shot) | ~20 min |
| 3 | `crear_indice.py` | Une Food-101 (descargar aparte en `datos/food-101`) y las fotos limpias; split train/test | segundos |
| 4 | `extraer_embeddings.py --ruido-train-web 1` | Vector de 768 números por foto y por su copia con ruido | ~4 h |
| 5 | `entrenar_clip.py` | Entrena el clasificador, calibra la certeza y evalúa | ~2 min |
| 6 | `evaluar_v1.py` | Mide el modelo original con el mismo test (comparación justa) | ~15 min |
| 7 | `prueba_profe.py descargar` / `evaluar` | Fotos nuevas + 6 tipos de ruido en 3 niveles | ~20 min |

Food-101: `https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz` (5 GB), descomprimir en `datos/`.

## Cómo funciona

1. **Características:** SigLIP2 (sin entrenarlo) convierte la foto en un vector de 768 números.
2. **Platillo:** una capa lineal (regresión logística multinomial) da la probabilidad de cada uno de los 146 platillos. Se divide entre una temperatura `T` calibrada para que "80 % de certeza" acierte ~80 % de las veces.
3. **Sabores:** `P(sabor) = P(platillo) @ M`, donde `M` es la tabla platillo → sabor (`catalogo.py`). Un sabor se marca si pasa de 50 %.
4. **Ruido:** cada foto de entrenamiento se usa limpia y con una copia degradada al azar (`ruido.py`), así el modelo aprende a reconocer el platillo aunque la foto venga mal.

## Estructura

```
├── app.py                    # Interfaz web (Gradio)
├── catalogo.py               # 146 platillos: sabores, nombre en español y búsquedas
├── comun.py                  # Carga de SigLIP2 y utilidades
├── ruido.py                  # Generador de ruido (entrenamiento y prueba controlada)
├── descargar_fotos.py        # Paso 1
├── limpiar_fotos.py          # Paso 2
├── crear_indice.py           # Paso 3
├── extraer_embeddings.py     # Paso 4
├── entrenar_clip.py          # Paso 5
├── evaluar_v1.py             # Paso 6
├── prueba_profe.py           # Paso 7
├── comparar_extractores.py   # Comparación de extractores (por qué SigLIP2)
├── modelo_sabores_v2.npz     # Modelo entrenado (sin pickle: solo arreglos numpy)
├── pesos/                    # Pesos de visión de SigLIP2 en fp16 (186 MB, Apache 2.0; solo en el zip)
├── resultados/               # Métricas en JSON/CSV
├── ejemplos/                 # Fotos de ejemplo para la app
├── docs/                     # Reporte (docx y pdf)
├── entrenar.py, entrenar_platillo.py, entrenar_embeddings.py, extraer_embeddings_v1.py, crear_csv.py, utils.py
│                             # Scripts de la v1 (se conservan como referencia)
└── modelo_embeddings_A.joblib  # Modelo v1 (para la comparación)
```

## Limitaciones

- Solo conoce 146 platillos: si le das otro, contesta el más parecido (por eso existe el aviso de "no estoy seguro").
- La etiqueta de sabor es por platillo, no por foto (unos tacos sin salsa siguen marcados como picantes).
- Las fotos de internet pueden tener errores de etiqueta aunque se filtraron.

## Licencias

- SigLIP2 (Google): Apache 2.0. open_clip, timm, Gradio, scikit-learn, PyTorch, icrawler: licencias permisivas (MIT/Apache/BSD).
- Food-101: uso para investigación. Las fotos descargadas de internet no se incluyen en el repositorio.

## Equipo

Abarca Perez Daniel Isaac · Gómez Gómez Ernesto · Guerrero Pérez Paola Guadalupe · Mendoza Paez Alejandro · Olvera De La Huerta Santiago · Pérez Salinas Juan Jesús

Machine Learning — ESCOM, IPN — Grupo 6CM4
