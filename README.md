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
- `delete-old-vs-files` / `delete-old-vectors`
- `orchestrate`

## Versioning & Orchestration

### Layout de versiones

El layout canónico ahora es:

- `Biblioteca/vN/` (una carpeta por versión)
- `Biblioteca/vN/_state/library.json` (manifest de esa versión)
- `Biblioteca/archives/` (zips de versiones archivadas)

> Cambio relevante: el root de versión se movió de `Biblioteca/biblioteca/vN` a `Biblioteca/vN`.

### Resolución de la última versión

Toda resolución automática de manifest usa `worklib/versioning.py`:

- `get_biblioteca_root()`
- `list_versions()`
- `latest_version()`
- `latest_manifest_path()`
- `resolve_manifest_path(manifest_arg)`

Si no se pasa `--manifest`, se usa `resolve_manifest_path(None)` y por defecto toma `Biblioteca/v{latest}/_state/library.json`.

### Política de archivo de versiones

Al finalizar `orchestrate` con éxito (rebuild + fill + smoke test):

1. Se identifica la versión inmediatamente anterior.
2. Se crea zip en `Biblioteca/archives/vN_YYYYMMDD_HHMMSS.zip`.
3. Por defecto se elimina la carpeta anterior para ahorrar disco.

Flags:

- `--no-archive`: no zippea ni elimina.
- `--keep-old-folder`: zippea pero conserva la carpeta vieja.

### Orchestrate y borrado de vectores

`orchestrate` ya **no borra** archivos viejos de vector stores automáticamente.

- Para ejecutar ese borrado manualmente en `orchestrate`, agrega `--delete-old-vectors`.
- También puedes ejecutar explícitamente el subcomando:

```bash
python run_library_ops.py delete-old-vectors --manifest /ruta/al/manifest_viejo.json
```

### Smoke test post-orquestación

Por defecto, `orchestrate` ejecuta un smoke test E2E al final con el query:

- `¿qué es el SIAR?`

El smoke test usa el pipeline normal de consulta (pick/confirm/refine/retrieve/write) contra el manifest más reciente. Si falla, `orchestrate` termina con código no-cero.

Flags:

- `--smoke-test` (default ON)
- `--no-smoke-test`
- `--smoke-test-query "..."`

Ejemplo:

```bash
python run_library_ops.py orchestrate --apply
```

## Dónde vive el motor RAG y CLI actual

- Motor de consulta RAG: `worklib/query/pipeline.py` (`pro_query`).
- Motor de ingesta: `worklib/ingest.py` (`ingest_document` + CLI `main`).
- Entrypoints CLI existentes (sin cambios de uso):
  - `ingest_auto.py` → `worklib.ingest.main`
  - `query_auto.py` → `worklib.query.main`
  - `run_library_ops.py` / `python -m library_ops` → `library_ops.cli.main`

## Configuración de manifest y directorios (env vars)

La configuración base se resuelve con `worklib/versioning.py` y `worklib/config.py`.

- `WORKLIB_BIBLIOTECA_ROOT`: root explícito de `Biblioteca`.
- `WORKLIB_ROOT`: fallback alterno.
- `WORKLIB_MANIFEST_PATH` o `RAG_MANIFEST_PATH`: override directo de manifest.

Sin overrides, la ruta efectiva de manifest se resuelve a la última versión disponible (`Biblioteca/vN/_state/library.json`).

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
