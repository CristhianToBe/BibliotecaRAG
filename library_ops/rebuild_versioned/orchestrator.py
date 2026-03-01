from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from worklib.config import default_config
from worklib.store import save_manifest

from library_ops.openai_utils import get_client

from .llm_steps import get_prompts
from .utils import model_names
from .stages.bootstrap import stage_bootstrap_manifest, stage_load_manifest
from .stages.cleanup import stage_cleanup_empty_dirs
from .stages.manifest_build import stage_build_new_manifest
from .stages.materialize import stage_materialize_files
from .stages.plan import stage_build_final_paths, stage_generate_plan
from .stages.prefixes import stage_infer_prefixes
from .stages.proposals import stage_propose_base_paths
from .stages.taxonomy import stage_build_taxonomy, stage_write_prefixed_taxonomy_view
from .stages.validation import stage_validate_with_reproposal
from .stages.vectorstores import stage_create_vector_stores
from .stages.workspace import stage_prepare_workspace


def run(
    *,
    manifest_override: Optional[Path],
    apply: bool,
    mode: str,
    debug: bool,
    batch_size: int,
    prefix_batch: int,
    max_rounds: int,
    max_docs: int,
    cleanup_empty: bool,
    create_vector_stores: bool,
) -> int:
    load_dotenv()
    cfg = default_config()

    manifest_path = manifest_override if manifest_override else cfg.manifest_path
    rc_boot = stage_bootstrap_manifest(manifest_path, cfg.library_dir)
    if rc_boot != 0:
        return rc_boot

    manifest = stage_load_manifest(manifest_path)
    docs_all = list(manifest.docs.values())

    client = get_client()
    prompts = get_prompts()
    model_tax, model_nano, model_mini = model_names()

    docs = docs_all[: max_docs] if max_docs and max_docs > 0 else docs_all

    _, vroot, version_label, v_state, v_lib = stage_prepare_workspace(cfg.library_dir)

    if debug:
        print(f"[debug] version_root={vroot}")
        print(f"[debug] apply={apply} mode={mode} docs={len(docs)}")

    doc_prefix = stage_infer_prefixes(
        client,
        model_nano,
        docs,
        prefix_batch,
        prompts["prefix"],
        prompts["prefix_normalize"],
        debug=debug,
    )

    _, base_paths = stage_build_taxonomy(client, model_tax, manifest, version_label, vroot, taxonomy_system=prompts["taxonomy"])
    stage_write_prefixed_taxonomy_view(vroot, doc_prefix, base_paths)

    proposals, tmp_vs = stage_propose_base_paths(client, model_nano, manifest, docs, base_paths, propose_system=prompts["propose"])

    accepted_base = stage_validate_with_reproposal(
        client,
        model_mini,
        model_nano,
        manifest,
        docs,
        proposals,
        base_paths,
        validate_system=prompts["validate"],
        propose_system=prompts["propose"],
        tmp_vs=tmp_vs,
        batch_size=batch_size,
        max_rounds=max_rounds,
    )

    accepted_final, final_paths = stage_build_final_paths(docs, doc_prefix, accepted_base)
    moves = stage_generate_plan(version_label, vroot, v_lib, docs, accepted_final)

    print("\n=== RESUMEN ===")
    print(f"- versión nueva: {version_label} -> {vroot}")
    print(f"- docs procesados: {len(docs)}")
    print(f"- base_paths: {len(base_paths)}")
    print(f"- paths finales (autor/año/base): {len(final_paths)}")
    print(f"- movimientos: {len(moves)} (mode={mode})")
    print(f"- apply: {apply}")

    if not apply:
        print(f"📝 No se aplicó nada. Revisa {vroot/'plan.json'} y {vroot/'taxonomy_prefixed.txt'}.")
        if debug:
            print("[debug] dry-run implícito por apply=False")
        return 0

    stage_materialize_files(moves, mode)

    new_manifest = stage_build_new_manifest(manifest, docs, final_paths, accepted_final, v_lib)

    if create_vector_stores:
        stage_create_vector_stores(client, version_label, new_manifest, docs, accepted_final)

    # mantener comportamiento original: docs ya construidos antes de vector stores,
    # por lo que vector_store_id de docs no se rellena incluso si existen VS nuevos.
    out_manifest = v_state / "library.json"
    save_manifest(out_manifest, new_manifest)
    print(f"✅ Nuevo library.json: {out_manifest}")

    if mode == "move" and cleanup_empty:
        stage_cleanup_empty_dirs(cfg.library_dir)

    print("\n✅ Listo. La nueva versión quedó aislada; el humano decide si borra versiones anteriores.")
    return 0
