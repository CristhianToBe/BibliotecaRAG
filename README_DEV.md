# README_DEV (Windows)

Guía rápida para desarrollo local de la web app (backend FastAPI + frontend estático).

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

## 4) Ejecutar backend

```powershell
uvicorn webapp.backend.app:app --reload --host 127.0.0.1 --port 8000
```

Abre: `http://127.0.0.1:8000/`

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
