# Credential output fixture

Synthetic fixtures from `/realm/inbox/spec`, commit `a51a65589d79dd5d11b1dfcf733299b5b2c809dd`:

- `polylogue-sinex-joint/prototype/masking/catalog.json`, SHA-256 `83a32a09b5fd29e1493a220770d57a02054f691797bec72f96a0fdba3444a50e`, copied unchanged.
- `polylogue-sinex-joint/prototype/masking/conformance.json`, SHA-256 `4094a0cb52807afed876b48bb32e3814817bfb0eeabd6aedeb8f04915dd991b1`.

The policy is `polylogue-sinex-joint/credential-masking.md` at that commit. The 132 conformance cases and their UTF-8 byte regions were written by hand, not produced by the masker. Each text is stored as `text_parts` of at most eight characters, joined by the test, so the source holds no credential-shaped literals; the joined texts match the source corpus exactly. Update these files only from a reviewed revision of the policy and corpus.

The test replaces each marker with a placeholder and compares the surrounding text with the expected regions.
