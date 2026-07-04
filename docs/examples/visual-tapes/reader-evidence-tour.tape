# Browserless local reader evidence lane against synthetic fixtures
# Auto-generated from visual evidence spec: reader-evidence-tour

Output reader-evidence-tour.gif

Set FontSize 16
Set Width 1320
Set Height 680
Set Padding 20

Type "devtools lab smoke run reader-visual-smoke --json --report-dir reader-evidence-tour"
Enter
Sleep 5s
Type "python -m json.tool reader-evidence-tour/reader-visual-smoke.json | sed -n '1,80p'"
Enter
Sleep 2s

# Let output settle
Sleep 1s
