import glob
import json
import os

OUT = "/tmp/claude-1000/-realm-project-polylogue/900c6128-99a3-4986-8ed4-d5d7aacb38fe/scratchpad"
files = sorted(glob.glob(f"{OUT}/findings_*.json"))
cats = ["underspecified", "gaps", "inconsistencies", "dep_priority", "reorg"]
agg = {c: [] for c in cats}
present = []
for f in files:
    dom = os.path.basename(f)[9:-5]
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception as e:
        print(f"  !! parse fail {dom}: {e}")
        continue
    present.append(dom)
    for c in cats:
        for item in d.get(c) or []:
            item["_dom"] = dom
            agg[c].append(item)
print(f"FINDINGS FILES: {len(present)}/16 -> {', '.join(present)}")
print(f"MISSING: {16 - len(present)}")
print()
for c in cats:
    print(f"=== {c.upper()} ({len(agg[c])}) ===")
with open(f"{OUT}/AGG.json", "w") as fh:
    json.dump(agg, fh, indent=1)
# quick per-category id lists for dedup
print("\n--- reorg proposals (id -> change) ---")
for r in agg["reorg"]:
    print(
        f"  [{r.get('_dom')}] {r.get('id', '?'):16} {r.get('change', '?'):12} {str(r.get('from', '')):8}->{str(r.get('to', ''))[:40]}"
    )
print("\n--- gaps (proposed new beads) ---")
for g in agg["gaps"]:
    print(
        f"  [{g.get('_dom')}] P{g.get('priority', '?')} {g.get('type', '?'):8} parent={str(g.get('parent', '')):14} {g.get('proposed_title', '')[:55]}"
    )
