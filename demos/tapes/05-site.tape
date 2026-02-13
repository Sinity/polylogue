# Site tape — generates static HTML archive

Output demos/output/05-site.gif

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

# Build static site
Type "polylogue site -o /tmp/polylogue-demo/site"
Sleep 400ms
Enter
Sleep 4s

# Show what was generated
Type "ls /tmp/polylogue-demo/site/"
Sleep 400ms
Enter
Sleep 2s

Sleep 2s
