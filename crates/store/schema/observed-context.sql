-- Explicit source role for newly declared observed inputs. Immutable imports
-- keep their existing original_hash/context_json tuple and CHECK unchanged.
CREATE TABLE observed_context (
  input         TEXT PRIMARY KEY REFERENCES input(id),
  version       INTEGER NOT NULL CHECK (version = 1),
  context_json  TEXT NOT NULL CHECK (json_valid(context_json) AND json_type(context_json) = 'array')
) STRICT, WITHOUT ROWID;

-- Companion ownership is an exact context value, independent of the number of
-- unrelated observed inputs. Keep this expression aligned with manifest().
CREATE INDEX input_claude_tool_result_main
  ON input(json_extract(context_json, '$[0]'))
  WHERE format = 'claude-code-tool-result';
