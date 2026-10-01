#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["SMOKE_RECEIPT"], "a") as output:
    print(json.dumps({"event": "cli", "argv": sys.argv[1:]}), file=output)
if os.environ.get("SMOKE_FAILURE") == "query" and "analyze" in sys.argv:
    raise SystemExit(3)
