from __future__ import annotations

from typing import Any, Dict, List

from .llm import call_text, eprint, safe_get, MODEL_FAST

from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_file_search_results(response_obj: Any, *, debug: bool = False) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    out_items = getattr(response_obj, "output", None) or []
    if debug:
        eprint("\n[DEBUG] response.output item types:", [getattr(x, "type", None) for x in out_items])

    for item in out_items:
        if getattr(item, "type", None) != "file_search_call":
            continue
        item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
        tool_results = (
            item_dict.get("results")
            or item_dict.get("search_results")
            or item_dict.get("file_search", {}).get("results")
        )
        if tool_results:
            if isinstance(tool_results, list):
                results.extend(tool_results)
            elif isinstance(tool_results, dict) and isinstance(tool_results.get("results"), list):
                results.extend(tool_results["results"])

        if debug:
            eprint("\n[DEBUG] file_search_call item (resumen):")
            eprint("  id:", item_dict.get("id"))
            eprint("  status:", item_dict.get("status"))
            eprint("  queries:", item_dict.get("queries"))
            eprint("  has_results:", bool(tool_results))

    return results


def normalize_result_text(res: Dict[str, Any]) -> str:
    if isinstance(res.get("content"), str):
        return res["content"]
    if isinstance(res.get("text"), str):
        return res["text"]
    content = res.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
        return "\n".join([x for x in parts if x.strip()])
    return ""


def retrieve_via_tool(
    vector_store_ids: List[str],
    query: str,
    *,
    max_num_results: int = 12,
    debug: bool = False,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Hace retrieval contra N vector stores, respetando el límite de 2 por llamada.
    Ejecuta las llamadas en paralelo (fan-out) y luego mergea resultados (fan-in).
    """

    # dividir en grupos de máximo 2 vector stores
    chunks = [vector_store_ids[i:i + 2] for i in range(0, len(vector_store_ids), 2)]
    if not chunks:
        return []

    # worker para 1 chunk
    def _run_chunk(vs_chunk: List[str]) -> List[Dict[str, Any]]:
        tools = [{
            "type": "file_search",
            "vector_store_ids": vs_chunk,
            "max_num_results": max(1, min(50, int(max_num_results))),
        }]

        resp = call_text(
            MODEL_FAST,
            system="Eres un recuperador. Llama file_search con la consulta del usuario.",
            user=query,
            tools=tools,
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            debug=debug,
        )

        raw_results = extract_file_search_results(resp, debug=debug)

        normed: List[Dict[str, Any]] = []
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            text = normalize_result_text(r)
            normed.append({
                "file_id": r.get("file_id") or safe_get(r, "document", "file_id") or r.get("document_id"),
                "filename": r.get("filename") or safe_get(r, "document", "filename"),
                "score": r.get("score") or r.get("relevance_score") or r.get("rank") or None,
                "text": text,
                "raw": r,
            })

        # filtrar vacíos
        return [
            x for x in normed
            if (x.get("file_id") or x.get("filename")) and (x.get("text") or "").strip()
        ]

    # ejecutar en paralelo
    all_results: List[Dict[str, Any]] = []
    workers = max(1, min(int(max_workers), len(chunks)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_chunk, ch): ch for ch in chunks}
        for fut in as_completed(futs):
            ch = futs[fut]
            try:
                all_results.extend(fut.result())
            except Exception as e:
                # No tumbamos toda la consulta por 1 chunk fallido
                if debug:
                    eprint(f"[WARN] retrieve chunk failed vs_ids={ch}: {type(e).__name__}: {e}")

    # dedupe (file_id + inicio del texto) preservando orden de llegada
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for h in all_results:
        key = (h.get("file_id") or "", h.get("filename") or "", (h.get("text") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    if debug:
        eprint(f"\n[DEBUG] retrieve_via_tool: chunks={len(chunks)} workers={workers} results={len(deduped)}")

    return deduped