# Overview tape — shows `polylogue` (no args) → stats panels
# Generated demo data is pre-seeded via scripts/seed_demo.py

Output demos/output/01-overview.gif

Set Theme "Catppuccin Mocha"
Set Width 1200
Set Height 600
Set FontSize 22
Set Padding 20
Set TypingSpeed 60ms

# Hide the setup phase
Hide
Type "source /tmp/polylogue-demo/.env"
Enter
Sleep 500ms
Show

# Show the polylogue overview (no arguments = stats)
Type "polylogue"
Sleep 400ms
Enter
Sleep 4s

# Let the user see the stats output
Sleep 2s
