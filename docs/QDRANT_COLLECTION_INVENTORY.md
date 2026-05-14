# Qdrant Collection Inventory & Migration State

**Audited:** 2026-05-14  
**Qdrant host:** `http://100.70.251.20:6333` (Ivan, Tailscale)  
**Current embedding model:** `mxbai-embed-large` (1024-dim, via Ollama on `http://100.79.3.15:11434`)  
**Authoritative tool for collection ops:** `mcp__tos-bridge__manage_collections` (since v0.5.0)

---

## TL;DR

The Qdrant instance carries the scars of an embedding-model migration that
was never finished. **86 of 119 collections still use 384-dim vectors**
(legacy `all-minilm:l6-v2`), **30 use the current 1024-dim**, **1 uses
1536-dim** (likely OpenAI text-embedding-3-small), **2 use 4096-dim**
(likely SmythOS test artefacts). When the model switched, parallel `_v2`
or `_1024` collections were created alongside the originals. The originals
were left in place.

The most important consequence: tos-bridge's `sync_to_tos` defaulted to
the legacy `ebatt_pattern_library` (384-dim) until v0.5.1 (2026-05-14).
Every default-arg call has been silently failing on the Qdrant side for
the past ~6 months, leaving Neo4j Pattern node count (165) significantly
ahead of Qdrant point count (56). **Fixed in tos-bridge commit `b0d92fb`
by changing the default to `ebatt_patterns_v2` (1024-dim, 61 points).**

---

## Dimension distribution

|  Dim | Collections | Notes                                                                                                               |
| ---: | ----------: | ------------------------------------------------------------------------------------------------------------------- |
|  384 |          86 | Legacy `all-minilm:l6-v2`. Mostly stale; some still receiving writes from code that hard-codes the collection name. |
| 1024 |          30 | Current `mxbai-embed-large`. Active.                                                                                |
| 1536 |           1 | `unified_pattern_library` — likely OpenAI text-embedding-3-small era.                                               |
| 4096 |           2 | `smythos_agent_memory` (0 pts), `smythos_skill_tests` (1 pt). Test/experimental.                                    |

## Active 1024-dim collections (30 total)

The collections aligned with the current model. Safe targets for new writes.

| Collection                          | Points | Likely owner / purpose                                       |
| ----------------------------------- | -----: | ------------------------------------------------------------ |
| `iwo_pipeline_history`              |   3011 | iwo-dbos pipeline events (largest)                           |
| `project_memory_v2`                 |    479 | Cross-project memory                                         |
| `system_configuration`              |    373 | Vanya's system config records                                |
| `project_memory_1024`               |    361 | Predecessor to project_memory_v2                             |
| `vault_specs`                       |    323 | Spec corpus                                                  |
| `system_configuration_1024`         |    317 | Sibling of system_configuration                              |
| `ebatt_second_brain`                |     65 | eBatt knowledge base (uses named vector `dense`)             |
| `ebatt_patterns_v2`                 | **61** | **eBatt pattern library — sync_to_tos default since v0.5.1** |
| `memories_1024`                     |     60 | Generic memory bucket                                        |
| `ethospower_ai_memory`              |     39 | EthosPower memory                                            |
| `mcp_server_configs`                |     28 | MCP server documentation                                     |
| `erpnext_configs`                   |     29 | ERPNext config snippets                                      |
| `jwnz`                              |     26 | JWNZ project                                                 |
| `ebatt-ai-memories`                 |     21 | eBatt-specific memories                                      |
| `boris_workflow_skills`             |     17 | Boris/IWO skill descriptions                                 |
| `project-instructions-templates`    |     17 | Project init templates                                       |
| `blender_project_1024`              |     14 | Blender project knowledge                                    |
| `ebatt_marketing_vault`             |     12 | eBatt marketing content (uses `dense` named vector)          |
| `ethospower-memory`                 |     11 | EthosPower memory (uses `dense` named vector)                |
| `bug-fixes-new`                     |     10 | Newer bug-fix patterns                                       |
| `handoff_instructions_v2`           |     10 | Agent handoff instructions                                   |
| `selfless`                          |      8 | (unclear)                                                    |
| `ethospower_customer_portal`        |      6 | EthosPower customer portal                                   |
| `conversation_state_1024`           |      3 | Conversation state                                           |
| `iwo-handoffs`                      |      1 | iwo-dbos handoff state                                       |
| `tos_bridge_test`                   |      3 | tos-bridge smoke test residue (safe to clean)                |
| `ethospower_marketing_vault`        |      2 | EthosPower marketing                                         |
| `ethospower_second_brain`           |      3 | EthosPower brain                                             |
| `claude-code-project-setup-scripts` |      1 | Setup script snippets                                        |
| `doc_embed_sessions`                |      1 | Doc-embedding session log                                    |

## Legacy 384-dim collections (86 total)

These should generally be treated as **read-only museum pieces**. Listed here
the ones with non-trivial content (≥ 20 points) — anything smaller is almost
certainly stale or test residue.

| Collection                                    | Points | Notes                                                            |
| --------------------------------------------- | -----: | ---------------------------------------------------------------- |
| `ebatt_spec_alignment_project`                |    189 | Spec alignment work. Possibly worth migrating.                   |
| `appflowy_integration_guide`                  |    154 | AppFlowy docs.                                                   |
| `vps2_deployment_tasks`                       |    109 | VPS deployment notes.                                            |
| `all_specs_enhancement_and_alignment_project` |    107 | Spec enhancement project.                                        |
| `bug_fixes_v1`                                |     96 | Old bug-fix collection. Compare against `bug-fixes-new` (1024).  |
| `ebatt_token_optimization_system`             |     72 | TOS notes — ironic given tos-bridge is the modern successor.     |
| `vanya_system_config`                         |     58 | System config (different from `system_configuration` at 1024).   |
| `ebatt_pattern_library`                       |     56 | **Old default of sync_to_tos. Superseded by ebatt_patterns_v2.** |
| `shared_pattern_library`                      |     49 | Cross-project patterns.                                          |
| `architectural_decisions`                     |     43 | ADRs.                                                            |
| `architectural_decisions-...-2025-10-29-04`   |     39 | Dated snapshot of the above.                                     |
| `ebatt_spec_cache`                            |     39 | Spec cache.                                                      |
| `project_mcp_conventions`                     |     39 | MCP conventions per project.                                     |
| `usefulaiguy_website`                         |     38 | Web scrape.                                                      |
| `ebatt_integrated_enhancements`               |     37 | Integrated enhancements.                                         |
| `appflowy_mcp_knowledge`                      |     33 | AppFlowy MCP knowledge.                                          |
| `qdrant_memory_system`                        |     31 | Qdrant own-memory.                                               |
| `system_documentation`                        |     27 | System docs.                                                     |
| `n8n_workflow_patterns`                       |     25 | n8n patterns.                                                    |
| `sequential_thoughts_v1`                      |     22 | Sequential-thinking artefacts.                                   |
| `nextcloud_integration`                       |     21 | Nextcloud integration notes.                                     |
| `mcp_troubleshooting`                         |     20 | MCP debug log.                                                   |
| `msty_studio_fixes`                           |     20 | Msty Studio fixes.                                               |
| `project-mcp-bug-fixes`                       |     20 | Project MCP bug fixes.                                           |
| `taskmaster_plane_sync_architecture`          |     19 | TM+Plane sync architecture.                                      |
| `shared_integrated_enhancements.backup`       |     19 | Backup snapshot — likely safe to delete.                         |
| `cloudflare_ai_chat`                          |     19 | CF AI chat notes.                                                |

The remaining ~60 legacy collections have under 20 points each and are
mostly experimental/abandoned. They cost ~zero to keep but are noise in
`manage_collections list`.

## Outlier-dim collections

| Collection                |  Dim | Points | Suspected origin                                 |
| ------------------------- | ---: | -----: | ------------------------------------------------ |
| `unified_pattern_library` | 1536 |    145 | OpenAI text-embedding-3-small era (transitional) |
| `smythos_agent_memory`    | 4096 |      0 | SmythOS experiment                               |
| `smythos_skill_tests`     | 4096 |      1 | SmythOS experiment                               |

## Parallel-collection pairs (the "v2 / \_1024" pattern)

When the model switched, several domains spawned modern counterparts. The
old ones may or may not still be being written to by hard-coded paths in
external code. **If you find a tool writing to one of the legacy collections,
that's a bug-pattern equivalent to the sync_to_tos default we just fixed.**

| Domain                          | Legacy (384)                                                          | Modern (1024)                  | Risk                                                     |
| ------------------------------- | --------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------- |
| eBatt patterns                  | `ebatt_pattern_library` (56)                                          | `ebatt_patterns_v2` (61)       | **Fixed in tos-bridge v0.5.1.**                          |
| Bug fixes                       | `bug_fixes` (12), `bug_fixes_v1` (96)                                 | `bug-fixes-new` (10)           | New is much smaller — suspect old still receiving writes |
| Handoff instructions            | `handoff_instructions` (28)                                           | `handoff_instructions_v2` (10) | Old 3× bigger — likely still being written to            |
| Conversation state              | `conversation_state` (6)                                              | `conversation_state_1024` (3)  | Low traffic, low priority                                |
| Architectural decisions         | `architectural_decisions` (43) + dated snapshot                       | (no clean 1024 sibling)        | May still be the canonical store                         |
| Pattern library (cross-project) | `shared_pattern_library` (49), `unified_pattern_library` (145 @ 1536) | (no 1024 successor yet)        | Three generations — needs a consolidation decision       |
| Memory chains                   | `memory_chains` (0), `temporal_memories` (0)                          | `memories_1024` (60)           | Old ones empty — safe to delete                          |

## Named-vector vs unnamed-vector collections

Most collections use Qdrant's unnamed default vector field (called `_default`
when introspected). A few use named vectors:

- `ebatt_marketing_vault`, `ebatt_second_brain`, `ethospower-memory` —
  use named vector `dense` (1024). Likely created via a tool that explicitly
  configured named vectors.
- `ai_patterns`, `deployment_patterns`, `mcp_test`, `troubleshooting_patterns`,
  `workflow_patterns` — use named vector `fast-all-minilm-l6-v2` (384).
  This is FastEmbed's default named-vector convention.

The `_get_collection_vector_name` helper in `graph_tools.py` handles both
shapes automatically, so this is generally transparent to callers.

## sync_to_tos default-bug history

| Version    | Default collection      | Vector size | Status                                                                                                                                                         |
| ---------- | ----------------------- | ----------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ≤ v0.5.0   | `ebatt_pattern_library` |         384 | Silently failed all default-arg calls on Qdrant side since model switch (~2025-05). Neo4j writes succeeded — partial-success path returned `status="partial"`. |
| **v0.5.1** | `ebatt_patterns_v2`     |        1024 | **Correct.** Matches current `mxbai-embed-large` embedding output.                                                                                             |

How to detect a similar bug in your own code:

1. `sync_to_tos` returns `{"status": "partial", "neo4j": {...}, "qdrant": {"status": "error", ...}}`. Read it.
2. Or check `points_count` drift: if a Neo4j label has many more nodes than the corresponding Qdrant collection, suspect a partial-success going unnoticed.

## Recommended future maintenance (not done yet)

These are the obvious next-tier items if you ever want to keep going on
the audit:

1. **Audit external code for hard-coded collection names.** Grep `~/projects`,
   `~/Nextcloud/PROJECTS`, the n8n workflow JSONs etc. for any of the
   legacy 384-dim collection names. Each hit is a candidate `sync_to_tos`-style
   default-bug.
2. **Clean up obviously-junk collections.** `shared_integrated_enhancements.backup`,
   `coachgrow_integrated_enhancements` (0 pts), `coachgrow_spec_cache` (0 pts),
   `integration_failures` (0 pts), `memory_chains` (0 pts), `temporal_memories` (0 pts),
   `onesong_integrated_enhancements` (0 pts), `validation_patterns` (0 pts).
   Use `mcp__tos-bridge__manage_collections action=delete`.
3. **Decide canonical store for each domain** with parallel pairs (see table above)
   and either migrate legacy → modern or formally retire one.
4. **Re-embed migration script** if you ever want to revive a legacy collection.
   Pattern: pull Neo4j Pattern nodes filtered by `source`, call
   `manage_collections recreate` with 1024-dim, then iterate creating Qdrant
   points with `mcp__tos-bridge__store_doc_with_graph` (or write a direct
   qdrant-client batch).

## Files referenced

- `~/Nextcloud/PROJECTS/tos-bridge/src/tos_bridge/server.py` line 159 — the
  `sync_to_tos` default that was wrong until v0.5.1.
- `~/Nextcloud/PROJECTS/tos-bridge/src/tos_bridge/graph_tools.py` —
  collection-aware vector-name handling in `_get_collection_vector_name`.

## Cross-references in Neo4j memory

Captured 2026-05-14 by the audit run that produced this document:

- Entity `qdrant_collection_inventory_2026_05_14` (type: `AuditFinding`)
- Entity `ebatt_pattern_library_legacy` (type: `QdrantCollection`)
- Entity `ebatt_patterns_v2_active` (type: `QdrantCollection`)
- Relationship `ebatt_patterns_v2_active SUPERSEDES ebatt_pattern_library_legacy`
- Entity `tos_bridge_v0_5_1_sync_default_fix` (type: `BugFix`)

Search: `mcp__tos-bridge__find_entities query="qdrant_collection_inventory"`.
