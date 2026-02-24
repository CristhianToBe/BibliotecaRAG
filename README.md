# BibliotecaRAG

Repositorio para ingestar, organizar y consultar una biblioteca documental usando OpenAI + vector stores.

## Estructura

- `worklib/`: lógica base de ingestión y consulta (query pipeline, manifest, cliente OpenAI).
- `library_ops/`: operaciones administrativas sobre la biblioteca (rebuild versionado, completar vector stores, keywords, limpieza).
- `webapp/`: app web mínima (backend FastAPI + frontend Vue 3 estático).
- Wrappers de entrada rápida:
  - `ingest_auto.py`
  - `query_auto.py`
  - `run_library_ops.py`

## Requisitos

- Python 3.10+
- Dependencias en `requirements.txt`
- Variables de entorno en `.env` (al menos credenciales/modelos OpenAI según tu configuración)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

### Ingestar un documento

```bash
python ingest_auto.py /ruta/al/documento.pdf --copy
```

### Consultar

```bash
python query_auto.py "¿Cuál es la política de vacaciones?"
```

### Operaciones de biblioteca

```bash
python -m library_ops --help
# o
python run_library_ops.py --help
```

Subcomandos disponibles:

- `rebuild-versioned`
- `fill-vectorstores`
- `fill-keywords`
- `delete-old-vs-files`
- `orchestrate`

## Dónde vive el motor RAG y CLI actual

- Motor de consulta RAG: `worklib/query/pipeline.py` (`pro_query`).
- Motor de ingesta: `worklib/ingest.py` (`ingest_document` + CLI `main`).
- Entrypoints CLI existentes (sin cambios de uso):
  - `ingest_auto.py` → `worklib.ingest.main`
  - `query_auto.py` → `worklib.query.main`
  - `run_library_ops.py` / `python -m library_ops` → `library_ops.cli.main`

## Configuración de manifest y directorios (env vars)

La configuración base se resuelve en `worklib/config.py` con este orden:

- `WORKLIB_ROOT` (opcional)
- fallback a `./Biblioteca` junto a `worklib/config.py` (si existe)
- fallback final: `Path.cwd()`

Desde ese root:

- `WORKLIB_LIBRARY_DIR` (default: `<root>/biblioteca`)
- `WORKLIB_STATE_DIR` (default: `<root>/_state`)
- `WORKLIB_MANIFEST_PATH` (default: `<state_dir>/library.json`)

Para query, también se respeta `RAG_MANIFEST_PATH` como override de manifest.

## Web app mínima (FastAPI + Vue 3)

Arquitectura propuesta e implementada:

- Backend FastAPI (`webapp/backend/app.py`):
  - `GET /api/health`
  - `POST /api/chat`
  - `POST /api/upload`
- Frontend Vue 3 (`webapp/frontend/index.html`):
  - panel de chat
  - panel de upload
- Reuso directo de funciones Python (sin shelling-out):
  - chat llama `worklib.query.pipeline.pro_query`
  - upload llama `worklib.ingest.ingest_document`

### Ejecutar backend + frontend

```bash
uvicorn webapp.backend.app:app --reload --port 8000
```

Abrir `http://localhost:8000/`.

## Secretos y `.env`

- Mantén credenciales fuera de git.
- Usa `.env` local (ejemplo: `OPENAI_API_KEY`, modelos y paths).
- `python-dotenv` se carga tanto en CLI como en backend web.

## Manifest y persistencia

Los comandos de `library_ops` que modifican el manifest usan una persistencia centralizada:

- Si se sobreescribe el manifest original, se crea backup automático (`*.bak_<timestamp>`).
- Si se usa `--out`, escribe en el archivo de salida sin backup del origen.

## Notas de refactor

Este repo fue limpiado para reducir duplicación en:

- bootstrap del cliente OpenAI (`library_ops/openai_utils.py` delega en `worklib.openai_client`),
- persistencia de manifest (`library_ops/manifest_json.py`),
- wrappers CLI mínimos y consistentes.
