# Credential output fixture

These synthetic fixtures originate from `/realm/inbox/spec`, commit `a51a65589d79dd5d11b1dfcf733299b5b2c809dd`:

- `polylogue-sinex-joint/prototype/masking/catalog.json`, SHA-256 `83a32a09b5fd29e1493a220770d57a02054f691797bec72f96a0fdba3444a50e`.
- `polylogue-sinex-joint/prototype/masking/conformance.json`, SHA-256 `4094a0cb52807afed876b48bb32e3814817bfb0eeabd6aedeb8f04915dd991b1`.

The catalog is copied unchanged. Conformance texts are stored as ordered `text_parts` of at most eight Unicode scalar values and concatenated by the test, avoiding credential-shaped literals in public source. Reconstruction of all 132 cases, including every UTF-8 extent and other field, was compared exactly with the pinned source corpus before publication. This is a storage encoding change; test input bytes and authored expectations are unchanged.

The source policy is `polylogue-sinex-joint/credential-masking.md` at that commit. The catalog owns the 22 fixed family shapes. The 132 conformance texts and UTF-8 byte extents were authored independently of the native matcher; tests read these expected extents directly and never generate them through production recognition. Refresh these copies only with a reviewed policy/corpus revision.

Run the focused tests through the declared operation:

```sh
agentctl job start polylogue foundation_exec --workspace WORKTREE -- cargo test --locked -p credential-output
```

Native markers may differ from the reference's display syntax. Tests normalize native markers to a sentinel and compare exact surrounding text, extent ordering and marker count against the authored regions. Every single byte cut is decoded to complete UTF-8 scalar values before passing `&str` pieces to the API, keeping at most three undecoded transport bytes.
