# BibliotecaRAG

Repositorio para ingestar, organizar y consultar una biblioteca documental usando OpenAI + vector stores.

## Estructura

- `worklib/`: lógica base de ingestión y consulta (query pipeline, manifest, cliente OpenAI).
- `library_ops/`: operaciones administrativas sobre la biblioteca (rebuild versionado, completar vector stores, keywords, limpieza).
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

## Manifest y persistencia

Los comandos de `library_ops` que modifican el manifest usan una persistencia centralizada:

- Si se sobreescribe el manifest original, se crea backup automático (`*.bak_<timestamp>`).
- Si se usa `--out`, escribe en el archivo de salida sin backup del origen.

## Notas de refactor

Este repo fue limpiado para reducir duplicación en:

- bootstrap del cliente OpenAI (`library_ops/openai_utils.py` delega en `worklib.openai_client`),
- persistencia de manifest (`library_ops/manifest_json.py`),
- wrappers CLI mínimos y consistentes.
