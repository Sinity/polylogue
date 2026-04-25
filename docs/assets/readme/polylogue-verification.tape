Output docs/assets/readme/polylogue-verification.gif

Require bash
Require devtools
Require jq

Set Shell "bash"
Set Theme "TokyoNight"
Set FontSize 22
Set Width 1280
Set Height 720
Set Padding 24
Set Framerate 18
Set TypingSpeed 16ms
Set WindowBar Colorful
Set WindowBarSize 36

Hide
Type "exec bash --noprofile --norc"
Enter
Sleep 500ms
Type "cd /realm/project/polylogue"
Enter
Type "export PS1='$ '"
Enter
Type "clear"
Enter
Show
Type "bash docs/assets/readme/readme-demo.sh verification"
Enter
Sleep 8s
