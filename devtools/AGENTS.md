# Devtools

## Code Review Rules

- Findings here are P2 unless the defect lets unverified code merge, makes a
  required gate vacuous, or commits private material across the public
  boundary.
- Receipts and caches are keyed on declared inputs. Do not ask for filesystem
  enumeration (installed trees, example databases, executables) as a key.
- Judge a test by the anti-vacuity condition it names, not by whether it could
  be stricter.
- Flag a selection that can silently become a zero-test or corpus run.
