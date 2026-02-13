# Run tape — shows `polylogue run --preview` then `polylogue run`

Output demos/output/02-run.gif

Set Theme "Catppuccin Mocha"
Set Width 1200
Set Height 600
Set FontSize 22
Set Padding 20
Set TypingSpeed 60ms

Hide
Type "source /tmp/polylogue-demo/.env"
Enter
Sleep 500ms
Show

# Show the preview first
Type "polylogue run --preview"
Sleep 400ms
Enter
Sleep 3s

# Then run the actual pipeline
Type "polylogue run"
Sleep 400ms
Enter
Sleep 5s

Sleep 2s
