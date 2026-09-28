# Devtools

## Code Review Rules

- Receipts and caches are keyed on declared inputs. Do not ask for filesystem
  enumeration (installed trees, example databases, executables) as a key.
- Judge a test by the anti-vacuity condition it names, not by whether it could
  be stricter.
- Flag a selection that can silently become a zero-test or corpus run.
