# README_DEV (Windows)

Guía rápida para desarrollo local de la web app con servidor único (FastAPI sirve API + frontend).

## Quick start

Haz doble clic en `run_app.bat` desde la raíz del repositorio.

El launcher:

- activa `.venv` si existe,
- inicia el backend (`uvicorn`) en `127.0.0.1:8000`,
- abre `http://127.0.0.1:8000/` en el navegador (frontend servido por FastAPI).

## 1) Requisitos

- Windows 10/11
- Python 3.10+
- PowerShell

## 2) Crear entorno e instalar dependencias

```powershell
cd C:\ruta\a\BibliotecaRAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Configurar variables (sin secretos en git)

```powershell
Copy-Item .env.example .env
notepad .env
```

Completa al menos:

- `OPENAI_API_KEY`

Opcionalmente ajusta rutas y límites web:

- `WORKLIB_LIBRARY_DIR`
- `WORKLIB_MANIFEST_PATH`
- `RAG_MANIFEST_PATH`
- `WEBAPP_UPLOAD_ALLOWED_EXTS` (ej: `pdf,txt,doc,docx`)
- `WEBAPP_UPLOAD_MAX_BYTES` (ej: `26214400` = 25MB)
- `WEBAPP_UPLOAD_DIR` (default `uploads`)

## 4) Ejecutar app (API + frontend)

```powershell
uvicorn webapp.backend.app:app --reload --host 127.0.0.1 --port 8000
```

Health: `http://127.0.0.1:8000/api/health`

Frontend: `http://127.0.0.1:8000/`


> Recomendación: abre siempre `http://127.0.0.1:8000/` (frontend servido por FastAPI).
> Evita abrir `file://...` o un `http.server` separado salvo que configures `API_BASE` apuntando al backend.

## 5) Checks rápidos

```powershell
python -m py_compile (rg --files -g "*.py")
python -c "import webapp.backend.app as a; print(a.app.title)"
```

> Nota: el segundo comando requiere que `fastapi` esté instalado correctamente.

## 6) CLI existente (sin cambios)

```powershell
python ingest_auto.py C:\ruta\documento.pdf --copy
python query_auto.py "¿Cuál es la política de vacaciones?"
python -m library_ops --help
```

## Performance tuning

Variables clave para latencia de `/api/chat`:

- `WEBAPP_CHAT_TIMEOUT_S` (default `90`): timeout total por request.
- `WEBAPP_RETRIEVE_TIMEOUT_S` (default `20`): timeout por retrieval/categoría.
- `WEBAPP_ANSWER_TIMEOUT_S` (default `45`): timeout de generación de respuesta.
- `WEBAPP_MAX_CATEGORIES` (default `2`): máximo categorías a consultar.
- `WEBAPP_TOP_K` (default `8`): resultados por vector store.
- `WEBAPP_MAX_PROMPT_CHARS` (default `12000`): máximo de caracteres de evidencia enviados al modelo de respuesta.
- `WEBAPP_RETRIEVAL_CACHE_TTL_S` (default `1800`): TTL cache retrieval.
- `WEBAPP_PICK_CACHE_TTL_S` (default `600`): TTL cache pick.

Selección de modelos (latencia-optimized por defecto):

- `MODEL_PICK` (default `gpt-5-nano`)
- `MODEL_CONFIRM` (default `gpt-5-nano`)
- `MODEL_REFINE` (default `gpt-5-nano`)
- `MODEL_ANSWER` (default `gpt-5-mini`)
- `MODEL_ARBITRATE` (default = `MODEL_ANSWER`)

Debug:

- En `/api/chat` con `debug=true`, la respuesta incluye `trace_id`, timings por etapa, tiempos de retrieval por categoría y resumen de llamadas de modelo.
- En el frontend, **Categorías manuales** ahora funciona como autocomplete con chips: puedes escribir, filtrar y seleccionar múltiples categorías; el payload se envía en `manual_categories` como string separado por `+` (ej: `laboral+tributario`).


## 7) Verificación manual mínima de Picker/categorías

Pruebas rápidas recomendadas en la UI o por API:

- **Picker ON + categorías manuales vacías** => **OK** (no debe devolver 400).
- **Picker OFF + categorías manuales vacías** => **400** con `missing_minimum`.
- **Picker OFF + categorías manuales completas** => **OK**.
