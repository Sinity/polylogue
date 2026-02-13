# Search tape — shows query mode with filters

Output demos/output/03-search.gif

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

# Basic search
Type 'polylogue "error handling"'
Sleep 400ms
Enter
Sleep 3s

# Search with provider filter
Type 'polylogue "API design" -p claude'
Sleep 400ms
Enter
Sleep 3s

Sleep 2s
