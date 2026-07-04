# Query/read drilldown against the deterministic demo archive
# Auto-generated from visual evidence spec: query-tour

Output query-tour.gif

Set FontSize 16
Set Width 1320
Set Height 680
Set Padding 20

Type "polylogue demo seed --root query-tour/archive --force --with-overlays --format json"
Enter
Sleep 4s
Type "POLYLOGUE_ARCHIVE_ROOT=query-tour/archive polylogue find pytest then read --view messages --limit 3"
Enter
Sleep 3s
Type "POLYLOGUE_ARCHIVE_ROOT=query-tour/archive polylogue find pytest then analyze --facets"
Enter
Sleep 2s

# Let output settle
Sleep 1s
