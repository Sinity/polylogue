-- The declared context (source role) of each observed input. Imported inputs
-- keep theirs in input.original_hash and input.context_json instead.
CREATE TABLE observed_context (
  input         TEXT PRIMARY KEY REFERENCES input(id),
  version       INTEGER NOT NULL CHECK (version = 1),
  context_json  TEXT NOT NULL CHECK (json_valid(context_json) AND json_type(context_json) = 'array')
) STRICT, WITHOUT ROWID;

-- Finds the tool-result files of one Claude main file without scanning other
-- inputs. Keep this expression in step with manifest() in claude_bodies.rs.
CREATE INDEX input_claude_tool_result_main
  ON input(json_extract(context_json, '$[0]'))
  WHERE format = 'claude-code-tool-result';
