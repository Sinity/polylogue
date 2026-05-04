# Schema Semantic Annotations

Schema elements under `schemas/providers/<provider>/` carry
`x-polylogue-semantic-role` annotations that describe the semantic
role of each JSON path (e.g., `message_role`, `conversation_title`,
`message_timestamp`).

## Review process

1. Run schema inference: `devtools schema-generate <provider>`
2. Review generated annotations for correctness
3. Create or update `pins.json` in the provider directory to reject
   known-wrong annotations (see `claude-code/pins.json` for format)
4. Re-run inference — `select_best_roles()` will respect pin overrides
5. Promote reviewed annotations: `devtools schema-promote <provider>`
6. Verify: `python devtools/verify_schema_annotations.py`

## Verification

```bash
python devtools/verify_schema_annotations.py
```

Exits 0 when all shipped providers have at least one annotated element.
