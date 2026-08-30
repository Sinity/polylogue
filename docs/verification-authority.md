# Verification authority

The verifier has one semantic owner per concern. AgentCTL supplies the outer
execution contract; Polylogue keeps project meaning and evidence; pytest
children are only interpreted for the individual child process they run.

| Concern | Sole authority |
| --- | --- |
| Checkout identity, selection graph, pytest lanes | Devtools |
| Child-process interpretation | Pytest child mechanics |
| Scratch, cgroup, resource admission, deadline, cancellation | AgentCTL |
| Logs and generic result | AgentCTL |
| Semantic receipt, retention, diagnostics | Devtools |

Foreground `devtools test`, affected verification, quick verification, and
explicit complete verification remain usable without AgentCTL variables. They
produce Polylogue semantic receipts and do not claim AgentCTL admission or
recorded run evidence. Declared AgentCTL operations invoke the same semantic entrypoints;
their exact-head/workspace, resource, scratch, deadline, cancellation, and
generic result contracts are owned by the project descriptor and AgentCTL.

The local receipt intentionally does not mirror generic job lifecycle. An
AgentCTL job log/result may correlate with a semantic receipt, but process
placement, host scheduling, workspace lifecycle, and outer cancellation are
not inferred from a verifier exit code.
