# ===================================================================
#
#  JQ script to format a JSONL log file into a complete, readable
#  Markdown workflow, including local commands and metadata.
#
#  Version 4: Show Everything, Formatted Intelligently.
#
#  Usage:
#    cat your_log.jsonl | jq -f format_log.jq -r | glow -
#
# ===================================================================

# --- Helper Functions ---

# (Unchanged from previous version)
def to_code_block(language):
  if type == "string" then
    (fromjson? // null) as $parsed |
    if $parsed != null then
      "```\(language)\n" + ($parsed | @json) + "\n```"
    else . end
  elif type == "object" or type == "array" then
    "```\(language)\n" + (. | @json) + "\n```"
  else tostring end;

# Formats the standard user/assistant message content (arrays of objects)
def format_structured_content_md:
  .[] | 
  if .type == "text" then .text
  elif .type == "tool_use" then
    "> <details>\n" +
    "> <summary><strong><kbd>🛠️ TOOL USE</kbd> \(.name)</strong></summary>\n" +
    "> \n" +
    "> " + (.input | to_code_block("json") | gsub("\n"; "\n> ")) + "\n" +
    "> </details>"
  elif .type == "tool_result" then
    "> **<kbd>✔️ TOOL RESULT</kbd>** for `\(.tool_use_id)`\n" +
    "> \n" +
    "> " + (.content | to_code_block("json") | gsub("\n"; "\n> "))
  else . | to_code_block("json") end;

# --- Main Filter ---

# A single, powerful filter that handles all log entry types.
. as $line |
(
  # --- 1. Handle Metadata Entries ---
  if $line.isMeta == true then
    "> ℹ️ *System Meta: \($line.message.content)*"
  
  # --- 2. Handle Local Command Entries ---
  elif ($line.message.content | type) == "string" and ($line.message.content | test("<command-name>")) then
    ( $line.message.content |
      capture("<command-name>(?<name>[^<]+)</command-name>.*<command-message>(?<msg>[^<]*)</command-message>.*<command-args>(?<args>[^<]*)</command-args>"; "s")
    ) as $cmd |
    "> **<kbd>🖥️ LOCAL COMMAND</kbd>** `\($cmd.name)`\n" +
    (if $cmd.args != "" then "> \n> ```sh\n> \($cmd.msg) \($cmd.args)\n> ```" else "" end)

  # --- 3. Handle Local Command Stdout ---
  elif ($line.message.content | type) == "string" and ($line.message.content | test("<local-command-stdout>")) then
    ( $line.message.content |
      capture("<local-command-stdout>(?<out>.*)</local-command-stdout>"; "s")
    ) as $stdout |
    if $stdout.out == "" then
      "> *↳ Local stdout: (empty)*"
    else
      "> **↳ Local stdout:**\n" +
      "> ```\n" +
      "> \($stdout.out | gsub("\n"; "\n> "))\n" +
      "> ```"
    end

  # --- 4. Handle Standard User/Assistant Messages ---
  else
    "### 🕒 `\($line.timestamp)` | **\($line.type | ascii_upcase)**\n" +
    (
      if ($line.message.content | type) == "array" then
        $line.message.content | format_structured_content_md
      else
        $line.message.content # Simple string content
      end
    )
  end
) + "\n---\n"
