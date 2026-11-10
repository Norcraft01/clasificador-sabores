# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y [Semantic Versioning](https://semver.org/lang/es/).

## [2.0.0] - 2026-11-10
### Added
- Extractor SigLIP2 ViT-B/16 (open_clip, WebLI) congelado + clasificador lineal calibrado con temperatura.
- 45 platillos nuevos (39 mexicanos + refuerzos de amargo y ácido); catálogo en `catalogo.py` (146 platillos).
- Descarga (`descargar_fotos.py`) y limpieza automática zero-shot (`limpiar_fotos.py`) de fotos web.
- Entrenamiento con copias degradadas (`ruido.py`: gaussiano, sal y pimienta, desenfoque, JPEG, pixelado, iluminación, color).
- Evaluación comparativa v1 contra v2 (`evaluar_v1.py`) y prueba tipo profesor con ruido controlado (`prueba_profe.py`).
- Interfaz Gradio nueva (`app.py`) con certeza por sabor y opción de ensuciar la foto.
- Lanzadores `INICIAR.bat` / `iniciar.sh` que instalan uv y eligen torch CPU o GPU.

### Changed
- `extraer_embeddings.py` reescrito para SigLIP2 (la versión de la v1 queda en `extraer_embeddings_v1.py`).
- La tabla platillo → sabor de los 101 platillos de Food-101 se conserva sin cambios.

## [1.0.0]
- Versión original del equipo (rama `main`): ResNet50 + regresión logística sobre Food-101.
