# Browser-backed deterministic ChatGPT/Claude capture proof
# Auto-generated from visual evidence spec: browser-capture-tour

Output browser-capture-tour.gif

Set FontSize 16
Set Width 1320
Set Height 680
Set Padding 20

Type "devtools workspace dev-loop --browser-provider-smoke --json > browser-capture-tour.json"
Enter
Sleep 18s
Type "python - <<'PY'"
Enter
Type "import json"
Enter
Type "p=json.load(open('browser-capture-tour.json'))['browser_provider_smoke']"
Enter
Type "print('ok', p['ok'])"
Enter
Type "print('providers', p['provider_statuses'])"
Enter
Type "print('popup', p['popup_status'])"
Enter
Type "print('screenshot', p['artifacts'].get('popup_screenshot'))"
Enter
Type "print('artifact_refs', p['artifact_refs'])"
Enter
Type "PY"
Enter
Sleep 2s

# Let output settle
Sleep 1s
