# Browser-backed deterministic ChatGPT/Claude live-follow proof
# Auto-generated from visual evidence spec: browser-capture-tour

Output browser-capture-tour.gif

Set FontSize 16
Set Width 1320
Set Height 680
Set Padding 20

Type "devtools workspace dev-loop --isolated-ports --browser-provider-live-follow --json > browser-capture-tour.json"
Enter
Sleep 24s
Type "python - <<'PY'"
Enter
Type "import json"
Enter
Type "p=json.load(open('browser-capture-tour.json'))['browser_provider_live_follow']"
Enter
Type "print('ok', p['ok'])"
Enter
Type "print('providers', p['provider_statuses'])"
Enter
Type "print('archive_ok', p['archive_ok'])"
Enter
Type "print('api_ok', p['api_ok'])"
Enter
Type "print('session_id', p['session_id'])"
Enter
Type "print('api_messages', {k: v['message_count'] for k, v in p['api_messages'].items()})"
Enter
Type "PY"
Enter
Sleep 2s

# Let output settle
Sleep 1s
