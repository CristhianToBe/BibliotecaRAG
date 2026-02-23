from pathlib import Path
import os

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "prompts"))

def load_prompt(name: str) -> str:
    """
    Carga un prompt desde prompts/<name>.txt
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt no encontrado: {path}")
    return path.read_text(encoding="utf-8")
