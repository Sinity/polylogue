Output docs/assets/readme/polylogue-quickstart.gif

Require devtools
Require polylogue
Require grep

Set Shell "bash"
Set Theme "Catppuccin Mocha"
Set FontSize 26
Set Width 1280
Set Height 720
Set Padding 24
Set Framerate 24
Set TypingSpeed 24ms

Hide
Type "cd /realm/project/polylogue"
Enter
Type "rm -rf .local/vhs-readme-demo"
Enter
Type "devtools lab-corpus seed --count 4 --output-dir .local/vhs-readme-demo --env-only 2>/dev/null | grep '^export ' > .local/vhs-readme-env.sh"
Enter
Type "source .local/vhs-readme-env.sh"
Enter
Type "export PS1='$ '"
Enter
Type "clear"
Enter
Show

Type "polylogue --plain stats"
Enter
Sleep 1800ms

Type "polylogue --plain error list --limit 4"
Enter
Sleep 1800ms

Type "clear"
Enter
Sleep 500ms

Type "polylogue --plain products profiles --limit 3"
Enter
Sleep 7s
