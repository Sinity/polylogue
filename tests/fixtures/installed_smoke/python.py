#!/usr/bin/env python3
import json
import os
import sys

if "-m" in sys.argv:
    with open(os.environ["SMOKE_RECEIPT"], "a") as output:
        print(json.dumps({"event": "module", "argv": sys.argv[1:]}), file=output)
else:
    print(os.environ["SMOKE_SOCKET"])
