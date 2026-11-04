# Backfill stored document bodies

tos-bridge before v0.6.0 embedded a document's text and then discarded it, so older points hold a title, a summary and a path. These scripts restore the text from Claude Code transcripts on the machine they run on. Run them on each machine whose transcripts might hold more.

```bash
export QDRANT_URL=... QDRANT_API_KEY=...
python3 scripts/backfill/match_transcripts.py --workdir ~/tos-backfill
```

This writes nothing to Qdrant. Read its counts, then snapshot each collection (`POST /collections/<name>/snapshots`). Keep `--workdir` out of any repository, because `plan.json` holds full document text.

```bash
python3 scripts/backfill/restore_bodies.py --workdir ~/tos-backfill --sample
```

The sample stores one document per collection and two with graph writes. Compare the payloads and each Document node's relationship count with their earlier values; only `synced_at` should change on existing fields. Then drop `--sample`. Run it again to retry failures.

`--wrapper` names a command that supplies the server's environment, for example a local secrets wrapper.

`match_transcripts.py` refuses text with a leaked tool envelope, text that looks like it holds a secret, and any call whose title or summary disagrees with the point. Refused points are listed in `refused.json` for a person to read. Points with no matching call are listed in `unmatched.json`.
