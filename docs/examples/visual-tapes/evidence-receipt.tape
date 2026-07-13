# Assistant claim checked against structural test outcomes
# Auto-generated from visual evidence spec: evidence-receipt

Output evidence-receipt.gif

Set FontSize 16
Set Width 1320
Set Height 680
Set Padding 20

Set TypingSpeed 0.02
Set CursorBlink false
Hide
Type "mkdir -p evidence-receipt && cd evidence-receipt && polylogue demo receipts --compact >/dev/null && printf 'READY\n'"
Enter
Wait+Screen /READY/
Type "clear"
Enter
Sleep 200ms
Show
Type "polylogue demo receipts --compact --no-seed"
Enter
Wait+Screen /verdict: contradicted_at_claim_time_then_repaired/
Sleep 4s

# Let output settle
Sleep 1s
