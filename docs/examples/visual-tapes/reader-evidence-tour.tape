# Browserless local reader evidence lane against synthetic fixtures
# Auto-generated from visual evidence spec: reader-evidence-tour

Output reader-evidence-tour.gif

Set Shell "bash"
Set FontSize 16
Set FontFamily "DejaVu Sans Mono"
Set Width 1320
Set Height 680
Set Padding 20

Hide
Type "devtools lab smoke run reader-visual-smoke --json --report-dir reader-evidence-tour >/dev/null 2>&1 && printf 'REA''DY\n'"
Enter
Wait+Screen@150s /READY/
Type "clear"
Enter
Sleep 200ms
Show
Type "python -m json.tool reader-evidence-tour/reader-visual-smoke.json | sed -n '1,80p'"
Enter
Sleep 2s

# Let output settle
Sleep 1s
