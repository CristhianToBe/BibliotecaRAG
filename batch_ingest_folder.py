from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def iter_files(folder: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_lower = {e.lower() for e in exts}
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower]
    else:
        files = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in exts_lower]
    return sorted(files)


def run_ingest(ingest_py: Path, file_path: Path, copy: bool, title_from_name: bool, author: str, tags: str) -> int:
    cmd = [sys.executable, str(ingest_py), str(file_path)]

    # title opcional
    if title_from_name:
        # título = nombre del archivo sin extensión
        cmd += ["--title", file_path.stem]

    # author opcional
    if author:
        cmd += ["--author", author]

    # tags opcional
    if tags:
        cmd += ["--tags", tags]

    if copy:
        cmd += ["--copy"]

    print(f"\n=== Ingestando: {file_path.name} ===")
    print("CMD:", " ".join(cmd))

    proc = subprocess.run(cmd, text=True, capture_output=True)

    # imprime TODO junto (evita mezclas)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Ejecuta ingest_auto.py sobre todos los archivos de una carpeta.")
    ap.add_argument("folder", help="Carpeta con documentos (PDF por defecto).")
    ap.add_argument("--ingest", default="ingest_auto.py", help="Ruta a ingest_auto.py (default: ./ingest_auto.py)")
    ap.add_argument("--recursive", action="store_true", help="Recorrer subcarpetas")
    ap.add_argument("--ext", default="pdf", help="Extensión a incluir (default: pdf). Puedes pasar varias separadas por coma: pdf,docx,txt")
    ap.add_argument("--copy", action="store_true", help="Pasa --copy a ingest_auto.py")
    ap.add_argument("--title-from-name", action="store_true", help="Pasa --title <nombre_archivo> a ingest_auto.py")
    ap.add_argument("--author", default="", help="Pasa --author a ingest_auto.py")
    ap.add_argument("--tags", default="", help="Pasa --tags a ingest_auto.py")
    ap.add_argument("--stop-on-error", action="store_true", help="Detenerse si un archivo falla")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    ingest_py = Path(args.ingest).resolve()

    if not ingest_py.exists():
        raise FileNotFoundError(f"No encontré ingest_auto.py en: {ingest_py}")
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"No es carpeta: {folder}")

    exts = [f".{e.strip().lstrip('.')}" for e in args.ext.split(",") if e.strip()]
    files = iter_files(folder, recursive=args.recursive, exts=exts)

    if not files:
        print("No encontré archivos con extensiones:", exts)
        return

    print(f"Encontrados {len(files)} archivos en {folder} (recursive={args.recursive})")

    ok = 0
    fail = 0

    for f in files:
        code = run_ingest(
            ingest_py=ingest_py,
            file_path=f,
            copy=args.copy,
            title_from_name=args.title_from_name,
            author=args.author,
            tags=args.tags,
        )
        if code == 0:
            ok += 1
        else:
            fail += 1
            print(f"❌ Falló: {f.name} (exit={code})")
            if args.stop_on_error:
                break

    print("\n========================")
    print("Resumen")
    print("========================")
    print("✅ OK:", ok)
    print("❌ FAIL:", fail)


if __name__ == "__main__":
    main()
