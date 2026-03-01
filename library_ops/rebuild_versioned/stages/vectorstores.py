from __future__ import annotations

from typing import Dict, List


def stage_create_vector_stores(client, version_label: str, new_manifest, docs, accepted_final: Dict[str, str]) -> None:
    print("\n🧠 Creando vector stores por categoría final...")
    vs_by_cat: Dict[str, str] = {}
    for cat_name in new_manifest.categories.keys():
        vs = client.vector_stores.create(name=f"{version_label}:{cat_name}")
        vs_by_cat[cat_name] = vs.id
        new_manifest.categories[cat_name].vector_store_id = vs.id

    cat_to_files: Dict[str, List[str]] = {}
    for d in docs:
        fp = accepted_final[d.doc_id]
        cat_name = fp.replace("/", "__")
        if d.openai_file_id:
            cat_to_files.setdefault(cat_name, []).append(d.openai_file_id)

    for cat_name, file_ids in cat_to_files.items():
        vsid = vs_by_cat.get(cat_name)
        if not vsid:
            continue
        for i in range(0, len(file_ids), 2000):
            batch_ids = file_ids[i : i + 2000]
            client.vector_stores.file_batches.create(vector_store_id=vsid, file_ids=batch_ids)
