# Query-Action Workflows

The executable `find QUERY then ACTION` product contract moved to the generated
product surface:

- [Executable Query-Action Workflows](../product/workflows.md)

That document is rendered from `polylogue/product/workflows.py` and the live CLI
action contracts, including context-pack and continuation handoff flows. It
powers the demo-archive golden-path tests for #2305/#2306. Edit the registry or
action contracts, then run:

```bash
devtools render product-workflows
devtools render product-workflows --check
devtools test tests/unit/product/test_query_action_workflows.py
```
