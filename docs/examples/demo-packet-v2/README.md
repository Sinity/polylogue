# Demo Packet v2 example

The smallest production-validated example is the registered
[`_packet-contract-stub`](../../../.agent/demos/_packet-contract-stub/).
The same validator also covers the public behavioral-archaeology packet and
the missing-evidence anti-demo; none is grandfathered.

A conforming packet:

1. cites at least one receipt from `claim.receipts`;
2. declares every cited ref once in the top-level `receipts` table;
3. binds every receipt to a packet-local artifact with `sha256`;
4. places the receipt ref in the bound artifact bytes;
5. uses the exact ordered `## Claim`, `## Corpus`, `## Method`, `## Findings`,
   `## Specimens`, `## Counterexamples`, `## Limits`, `## Non-claims`, and
   `## Reproduce` report headings;
6. keeps falsifier state consistent and control IDs and measurement names
   unique.

Run the same production gate used by repository verification:

```bash
devtools lab policy demo-packet-registry
```

The focused mutation suite removes the claim citation, receipt digest, and
canonical Claim heading, then introduces contradictory falsifier state and
duplicate identities. Each case names the production guard whose removal would
let that invalid packet pass.
