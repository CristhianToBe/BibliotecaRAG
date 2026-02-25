from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib import error, request


DEFAULT_QUERIES = [
    "¿Cuál es la política de vacaciones?",
    "¿Qué reglas aplican para retención en la fuente?",
    "¿Cómo reportar exógena?",
    "Resumen de obligaciones NIIF para pymes",
    "¿Qué plazos hay para IVA?",
]


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--queries-file", default="")
    args = ap.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [x.strip() for x in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if x.strip()]

    totals = []
    cache_hits = 0
    print("--- Benchmark /api/chat (debug=true) ---")
    for q in queries[:5]:
        t0 = time.perf_counter()
        try:
            status, body = post_json(f"{args.base_url}/api/chat", {"message": q, "debug": True})
            dt = time.perf_counter() - t0
            totals.append(dt)
            dbg = body.get("debug_info") or {}
            if body.get("status") == "ok" and dbg:
                if dbg.get("events"):
                    for evt in dbg.get("events"):
                        if evt.get("stage") == "retrieving" and evt.get("cache_hit"):
                            cache_hits += 1
                            break
            print(f"Q: {q[:50]}... status={status} total={dt:.2f}s stage_ms={dbg.get('timings_ms', {})}")
        except error.URLError as exc:
            print(f"Q: {q[:50]}... ERROR {exc}")

    median_s = statistics.median(totals) if totals else 0.0
    print("\n=== Summary ===")
    if totals:
        print(f"runs={len(totals)} median_total_s={median_s:.2f} min={min(totals):.2f} max={max(totals):.2f}")
    else:
        print("runs=0 (server unavailable)")
    print(f"cache_hit_rate={cache_hits}/{len(totals)} ({(100.0 * cache_hits / len(totals)) if totals else 0:.1f}%)")
    print("before_after: baseline_not_captured_in_repo -> see current timings above")


if __name__ == "__main__":
    main()
