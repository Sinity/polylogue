---
title: "Route inventory analysis"
source_url: "https://chatgpt.com/share/6a4d6219-0408-83eb-920b-d14d57b775c4"
conversation_id: "6a4d6219-0408-83eb-920b-d14d57b775c4"
shared_conversation_id: "6a4d6219-0408-83eb-920b-d14d57b775c4"
model: "gpt-5-5-pro"
message_count: 243
mapping_node_count: 244
create_time: 1783456281.099963
update_time: 1783456282.322308
---

# Route inventory analysis

## 1. system

[empty message]

## 2. system

[empty message]

## 3. system

[empty message]

## 4. user

Original custom instructions no longer available

## 5. user

_create_time: 1783433285.083_

[empty message]

## 6. tool

_create_time: 1783433287.425315_

[empty message]

## 7. tool

_create_time: 1783433289.7626722_

The output of this plugin was redacted.

## 8. assistant

_create_time: 1783433289.9431307_

[empty message]

## 9. assistant

_create_time: 1783433301.596021_

I’ll inspect the Polylogue source to enumerate daemon and capture receiver routes, then build a file-anchored route inventory, threat pass, fix-gap list, and negative-test matrix.

## 10. assistant

_create_time: 1783433313.760925_

[empty message]

## 11. assistant

_create_time: 1783433289.869904_

bash -lc ls -la /mnt/data && file /mnt/data/polylogue-all.tar\(83\).gz /mnt/data/05-analysis-daemon-attack-surface.md && du -h /mnt/data/polylogue-all.tar\(83\).gz /mnt/data/05-analysis-daemon-attack-surface.md

## 12. tool

_create_time: 1783433312.7808685_

The output of this plugin was redacted.

## 13. assistant

_create_time: 1783433312.780511_

bash -lc mkdir -p /mnt/data/polylogue_src && tar -xzf '/mnt/data/polylogue-all.tar(83).gz' -C /mnt/data/polylogue_src && find /mnt/data/polylogue_src -maxdepth 2 -type d | sed 's#^#/ #' | head -50 && find /mnt/data/polylogue_src -maxdepth 2 -type f | head -50

## 14. tool

_create_time: 1783433317.1639245_

The output of this plugin was redacted.

## 15. assistant

_create_time: 1783433323.231542_

[empty message]

## 16. assistant

_create_time: 1783433317.163498_

bash -lc mkdir -p /mnt/data/polylogue_work && tar -xzf /mnt/data/polylogue_src/polylogue/polylogue-working-tree.tar.gz -C /mnt/data/polylogue_work && find /mnt/data/polylogue_work -maxdepth 3 -type d | head -80 && find /mnt/data/polylogue_work -maxdepth 4 -type f | head -80

## 17. tool

_create_time: 1783433320.5002925_

The output of this plugin was redacted.

## 18. assistant

_create_time: 1783433324.456856_

[empty message]

## 19. assistant

_create_time: 1783433320.499878_

bash -lc cd /mnt/data/polylogue_work/polylogue && wc -l polylogue/daemon/http.py polylogue/browser_capture/receiver.py polylogue/browser_capture/server.py && sed -n '1,220p' polylogue/daemon/http.py

## 20. tool

_create_time: 1783433324.4416366_

The output of this plugin was redacted.

## 21. assistant

_create_time: 1783433328.776473_

[empty message]

## 22. assistant

_create_time: 1783433324.441228_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "_static_get_route|_parameterized_get_route|_StaticPostRoute|do_GET|do_POST|do_DELETE|RouteContract|route_contract" polylogue/daemon/http.py | head -200

## 23. tool

_create_time: 1783433327.6476564_

The output of this plugin was redacted.

## 24. assistant

_create_time: 1783433327.647243_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '220,380p' polylogue/daemon/http.py && sed -n '1220,1395p' polylogue/daemon/http.py

## 25. tool

_create_time: 1783433330.7737603_

The output of this plugin was redacted.

## 26. assistant

_create_time: 1783433336.310467_

[empty message]

## 27. assistant

_create_time: 1783433330.773376_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '1080,1235p' polylogue/daemon/http.py && sed -n '1235,1325p' polylogue/daemon/http.py

## 28. tool

_create_time: 1783433334.5251434_

The output of this plugin was redacted.

## 29. assistant

_create_time: 1783433337.612799_

[empty message]

## 30. assistant

_create_time: 1783433334.524618_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "class .*Handler|def __init__|_check_auth|access_token|Authorization|Host|Origin|client_address|_client_host|_api_host|log_message" polylogue/daemon/http.py polylogue/daemon/*.py polylogue/browser_capture/*.py browser-extension/src -S

## 31. tool

_create_time: 1783433338.001831_

The output of this plugin was redacted.

## 32. assistant

_create_time: 1783433338.001441_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '930,1080p' polylogue/daemon/http.py && sed -n '3600,3645p' polylogue/daemon/http.py

## 33. tool

_create_time: 1783433340.4231074_

The output of this plugin was redacted.

## 34. assistant

_create_time: 1783433344.534784_

[empty message]

## 35. assistant

_create_time: 1783433340.422705_

bash -lc cd /mnt/data/polylogue_work/polylogue && find . -path '*003_polylogue_kwsb_1.md' -o -path '*task_packets*' | head -50 && rg -n "kwsb|DNS|rebinding|admission|capture receiver|spool|host gate|Sec-Fetch|browser_capture|capture" prework-v2 polylogue docs tests .beads* 2>/dev/null | head -200

## 36. tool

_create_time: 1783433345.4332027_

The output of this plugin was redacted.

## 37. assistant

_create_time: 1783433348.765348_

[empty message]

## 38. assistant

_create_time: 1783433345.432716_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '1,180p' docs/security.md && sed -n '1400,1535p' polylogue/config.py

## 39. tool

_create_time: 1783433348.282815_

The output of this plugin was redacted.

## 40. assistant

_create_time: 1783433348.282182_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '1535,1575p' polylogue/config.py

## 41. tool

_create_time: 1783433350.7318578_

The output of this plugin was redacted.

## 42. assistant

_create_time: 1783433350.731412_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '1,260p' polylogue/daemon/route_contracts.py && sed -n '1,260p' polylogue/daemon/user_state_http.py && sed -n '1,260p' polylogue/daemon/workspace_routes.py

## 43. tool

_create_time: 1783433354.165303_

The output of this plugin was redacted.

## 44. assistant

_create_time: 1783433356.744668_

[empty message]

## 45. assistant

_create_time: 1783433354.164787_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def _routes|UserStateRoute\(|RouteContract\(" polylogue/daemon/user_state_http.py polylogue/daemon/route_contracts.py | sed -n '1,220p'

## 46. tool

_create_time: 1783433357.3189828_

The output of this plugin was redacted.

## 47. assistant

_create_time: 1783433357.318538_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '240,450p' polylogue/daemon/route_contracts.py && sed -n '580,635p' polylogue/daemon/user_state_http.py

## 48. tool

_create_time: 1783433359.5607362_

The output of this plugin was redacted.

## 49. assistant

_create_time: 1783433363.288686_

[empty message]

## 50. assistant

_create_time: 1783433359.560187_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '1,250p' polylogue/browser_capture/server.py && sed -n '250,450p' polylogue/browser_capture/server.py && sed -n '1,260p' polylogue/browser_capture/receiver.py && sed -n '260,700p' polylogue/browser_capture/receiver.py

## 51. tool

_create_time: 1783433363.8094537_

The output of this plugin was redacted.

## 52. assistant

_create_time: 1783433363.808931_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/server.py | sed -n '1,430p'

## 53. tool

_create_time: 1783433367.0815065_

The output of this plugin was redacted.

## 54. assistant

_create_time: 1783433370.716868_

[empty message]

## 55. assistant

_create_time: 1783433367.080882_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/receiver.py | sed -n '1,260p'

## 56. tool

_create_time: 1783433370.4091282_

The output of this plugin was redacted.

## 57. assistant

_create_time: 1783433370.408462_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/receiver.py | sed -n '260,430p'

## 58. tool

_create_time: 1783433373.228068_

The output of this plugin was redacted.

## 59. assistant

_create_time: 1783433373.22758_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/receiver.py | sed -n '430,660p'

## 60. tool

_create_time: 1783433376.0776048_

The output of this plugin was redacted.

## 61. assistant

_create_time: 1783433376.07708_

bash -lc cd /mnt/data/polylogue_work/polylogue && ls tests/unit/daemon | sed -n '1,200p' && sed -n '1,260p' tests/unit/daemon/test_daemon_http_security.py && sed -n '1,260p' tests/unit/daemon/test_daemon_http_contracts.py

## 62. tool

_create_time: 1783433380.562395_

The output of this plugin was redacted.

## 63. assistant

_create_time: 1783433384.685973_

[empty message]

## 64. assistant

_create_time: 1783433380.556696_

bash -lc cd /mnt/data/polylogue_work/polylogue && sed -n '260,620p' tests/unit/daemon/test_daemon_http_security.py

## 65. tool

_create_time: 1783433383.5808692_

The output of this plugin was redacted.

## 66. assistant

_create_time: 1783433387.769755_

[empty message]

## 67. assistant

_create_time: 1783433383.580282_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
import ast, re, pathlib
p=pathlib.Path('polylogue/daemon/http.py')
mod=ast.parse(p.read_text())
for node in mod.body:
    if isinstance(node, ast.ClassDef) and node.name=='DaemonAPIHandler':
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith('_handle') or isinstance(item, ast.FunctionDef) and item.name in ['_dispatch_get','do_GET','_do_post_impl','do_POST','_do_delete_impl','do_DELETE','_check_auth','_check_cross_origin','_check_shell_bootstrap_access','_serve_web_shell','_serve_paste_browser_page','_serve_attachment_library_page']:
                print(f'{item.name}: {item.lineno}')
PY

## 68. tool

_create_time: 1783433390.074246_

The output of this plugin was redacted.

## 69. assistant

_create_time: 1783433395.925027_

[empty message]

## 70. assistant

_create_time: 1783433390.07369_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/healthz.py | sed -n '1,220p' && nl -ba polylogue/daemon/metrics.py | sed -n '1,260p' && nl -ba polylogue/daemon/otlp_receiver.py | sed -n '1,220p'

## 71. tool

_create_time: 1783433394.3918562_

The output of this plugin was redacted.

## 72. assistant

_create_time: 1783433397.671138_

[empty message]

## 73. assistant

_create_time: 1783433394.391026_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def handle_metrics|polylogue_|archive|path|source|db|label|root|version|error|failure" polylogue/daemon/metrics.py | sed -n '1,260p'

## 74. tool

_create_time: 1783433398.3104343_

The output of this plugin was redacted.

## 75. assistant

_create_time: 1783433398.309834_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/metrics.py | sed -n '920,1180p'

## 76. tool

_create_time: 1783433400.857743_

The output of this plugin was redacted.

## 77. assistant

_create_time: 1783433406.241373_

[empty message]

## 78. assistant

_create_time: 1783433400.857185_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def handle_metrics" polylogue/daemon/metrics.py && nl -ba polylogue/daemon/metrics.py | sed -n '1540,1585p'

## 79. tool

_create_time: 1783433403.9651523_

The output of this plugin was redacted.

## 80. assistant

_create_time: 1783433403.964397_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/metrics.py | sed -n '1820,1865p'

## 81. tool

_create_time: 1783433407.0129528_

The output of this plugin was redacted.

## 82. assistant

_create_time: 1783433417.217811_

[empty message]

## 83. assistant

_create_time: 1783433407.012295_

bash -lc cd /mnt/data/polylogue_work/polylogue && find polylogue/rendering polylogue/daemon -maxdepth 2 -type f | sed -n '1,100p' && rg -n "escape\(|html.escape|markupsafe|innerHTML|insertAdjacentHTML|textContent|sanitize|DOMPurify|dangerously|raw|markdown|render|script|template" polylogue/rendering polylogue/daemon/*.py polylogue/daemon/web_shell* -S | sed -n '1,240p'

## 84. tool

_create_time: 1783433415.2799122_

The output of this plugin was redacted.

## 85. assistant

_create_time: 1783433422.383337_

[empty message]

## 86. assistant

_create_time: 1783433415.279282_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "function esc|escAttr|innerHTML|textContent|href|target|rel=|messageBlocksHtml|renderSessions|renderInspector|snippet|title|raw_preview|querySelector|dataset" polylogue/daemon/web_shell.py polylogue/daemon/web_shell_reader.py polylogue/daemon/web_shell_paste.py polylogue/daemon/web_shell_attachments.py polylogue/rendering -S | sed -n '1,300p'

## 87. tool

_create_time: 1783433420.5521643_

The output of this plugin was redacted.

## 88. assistant

_create_time: 1783433420.551539_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/web_shell.py | sed -n '400,430p;1060,1125p;1220,1320p;1860,1975p;2200,2265p' && nl -ba polylogue/daemon/web_shell_reader.py | sed -n '120,360p;400,455p'

## 89. tool

_create_time: 1783433424.2461576_

The output of this plugin was redacted.

## 90. assistant

_create_time: 1783433424.245387_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/web_shell_reader.py | sed -n '352,405p' && nl -ba polylogue/rendering/renderers/html_sanitizer.py | sed -n '1,220p' && nl -ba polylogue/rendering/renderers/html_messages.py | sed -n '1,260p'

## 91. tool

_create_time: 1783433427.6783206_

The output of this plugin was redacted.

## 92. assistant

_create_time: 1783433434.598529_

[empty message]

## 93. assistant

_create_time: 1783433427.677365_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/rendering/core.py | sed -n '1,260p' && nl -ba polylogue/rendering/core_markdown.py | sed -n '1,240p' && nl -ba polylogue/rendering/renderers/html.py | sed -n '1,220p' && nl -ba polylogue/rendering/renderers/html_template.py | sed -n '1,180p' && nl -ba polylogue/rendering/templates/session.html | sed -n '1,140p'

## 94. tool

_create_time: 1783433431.911469_

The output of this plugin was redacted.

## 95. assistant

_create_time: 1783433431.910816_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/rendering/templates/session.html | sed -n '180,280p' && nl -ba polylogue/rendering/core_messages.py | sed -n '1,220p' && nl -ba polylogue/rendering/renderers/html_highlighting.py | sed -n '1,260p'

## 96. tool

_create_time: 1783433434.9747918_

The output of this plugin was redacted.

## 97. assistant

_create_time: 1783433440.300597_

[empty message]

## 98. assistant

_create_time: 1783433434.974073_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/rendering/templates/session.html | sed -n '300,430p'

## 99. tool

_create_time: 1783433439.6646905_

The output of this plugin was redacted.

## 100. assistant

_create_time: 1783433446.960515_

[empty message]

## 101. assistant

_create_time: 1783433439.661711_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/rendering/blocks.py | sed -n '1,260p'

## 102. tool

_create_time: 1783433445.829082_

The output of this plugin was redacted.

## 103. assistant

_create_time: 1783433463.958946_

[empty message]

## 104. assistant

_create_time: 1783433445.779374_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/models.py | sed -n '1,260p' && rg -n "provider_session_id|session_id|native_id|id" polylogue/sources/parsers polylogue/archive polylogue/storage -S | sed -n '1,200p'

## 105. tool

_create_time: 1783433462.1944768_

The output of this plugin was redacted.

## 106. assistant

_create_time: 1783433466.020642_

[empty message]

## 107. assistant

_create_time: 1783433462.193738_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/identity.py | sed -n '1,160p' && nl -ba polylogue/sources/parsers/browser_capture.py | sed -n '130,220p'

## 108. tool

_create_time: 1783433465.7861257_

The output of this plugin was redacted.

## 109. assistant

_create_time: 1783433471.685063_

[empty message]

## 110. assistant

_create_time: 1783433465.785246_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "provider_session_id|native_id|session_id|ParsedSession|provider_message_id|message_id" polylogue/storage polylogue/archive polylogue/pipeline -S | sed -n '1,240p'

## 111. tool

_create_time: 1783433471.5423717_

The output of this plugin was redacted.

## 112. assistant

_create_time: 1783433471.541612_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/pipeline/ids.py | sed -n '1,90p'

## 113. tool

_create_time: 1783433474.5778143_

The output of this plugin was redacted.

## 114. assistant

_create_time: 1783433487.859563_

I’ve found the routing model: daemon dispatch is table-driven via `implemented_daemon_route_patterns()` and `user_state_http._routes()`, while receiver paths are explicit branches. The first concrete gap is missing central pre-dispatch enforcement, with GET, `/healthz/*`, `/metrics`, and OTLP bypassing normal token/Origin flow.

## 115. assistant

_create_time: 1783433489.210167_

[empty message]

## 116. assistant

_create_time: 1783433474.577116_

bash -lc cd /mnt/data/polylogue_src/polylogue && rg -n "kwsb|003_polylogue|DNS rebinding|Host gate|Sec-Fetch|spool governor|receiver token|browser-capture-auth|admission" . -S | sed -n '1,300p'

## 117. tool

_create_time: 1783433492.1512625_

The output of this plugin was redacted.

## 118. assistant

_create_time: 1783433643.684874_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "def _handle_provider_usage|access_token|client_disconnected|_handle_otlp_post|MAX_BROWSER_CAPTURE_BODY_BYTES|authToken|chrome\.storage|storage\." polylogue/daemon/http.py polylogue/daemon/events_http.py polylogue/browser-extension/src polylogue/browser_capture/server.py polylogue/browser_capture/receiver.py

## 119. tool

_create_time: 1783433647.1299987_

The output of this plugin was redacted.

## 120. assistant

_create_time: 1783433649.818657_

[empty message]

## 121. assistant

_create_time: 1783433647.129153_

bash -lc cd /mnt/data/polylogue_work/polylogue && find . -maxdepth 3 -type d | sed -n '1,120p' && rg -n "def _handle_provider_usage|access_token|client_disconnected|_handle_otlp_post" polylogue/daemon/http.py polylogue/daemon/events_http.py

## 122. tool

_create_time: 1783433650.238658_

The output of this plugin was redacted.

## 123. assistant

_create_time: 1783433650.237881_

bash -lc find /mnt/data/polylogue_work -maxdepth 4 -type d | grep -i 'extension\|browser' | head -80 && cd /mnt/data/polylogue_work/polylogue && rg -n "authToken|chrome\.storage|storage\.sync|storage\.local|browser_capture_auth|capture auth|receiver token" -S . | head -120

## 124. tool

_create_time: 1783433654.5183473_

The output of this plugin was redacted.

## 125. assistant

_create_time: 1783433657.908509_

[empty message]

## 126. assistant

_create_time: 1783433654.517334_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/http.py | sed -n '1030,1070p;1288,1355p;3408,3475p' && printf '\n--- events_http ---\n' && nl -ba polylogue/daemon/events_http.py | sed -n '1,260p' && printf '\n--- extension bg ---\n' && nl -ba browser-extension/src/background.js | sed -n '55,85p;170,190p' && printf '\n--- popup ---\n' && nl -ba browser-extension/src/popup.js | sed -n '295,320p;390,410p'

## 127. tool

_create_time: 1783433658.9921405_

The output of this plugin was redacted.

## 128. assistant

_create_time: 1783433658.991261_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "provider-usage|_provider_usage|provider_usage" polylogue/daemon/http.py polylogue -g'*.py' | sed -n '1,80p'

## 129. tool

_create_time: 1783433663.3674128_

The output of this plugin was redacted.

## 130. assistant

_create_time: 1783433663.366512_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/http.py | sed -n '228,245p;1800,1865p;1865,1935p'

## 131. tool

_create_time: 1783433667.3366363_

The output of this plugin was redacted.

## 132. assistant

_create_time: 1783433671.357462_

[empty message]

## 133. assistant

_create_time: 1783433667.335484_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "_handle_provider" -C 3 polylogue/daemon/http.py

## 134. tool

_create_time: 1783433670.6788123_

The output of this plugin was redacted.

## 135. assistant

_create_time: 1783433677.167688_

[empty message]

## 136. assistant

_create_time: 1783433670.677955_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/http.py | sed -n '1188,1238p;1238,1285p'

## 137. tool

_create_time: 1783433676.4856477_

The output of this plugin was redacted.

## 138. assistant

_create_time: 1783433684.886686_

[empty message]

## 139. assistant

_create_time: 1783433676.484213_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/route_contracts.py | sed -n '1,260p'

## 140. tool

_create_time: 1783433683.049539_

The output of this plugin was redacted.

## 141. assistant

_create_time: 1783433683.048435_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/route_contracts.py | sed -n '260,430p'

## 142. tool

_create_time: 1783433685.7265685_

The output of this plugin was redacted.

## 143. assistant

_create_time: 1783433685.725609_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/route_contracts.py | sed -n '430,520p' && printf '\n--- implemented patterns fn ---\n' && nl -ba polylogue/daemon/http.py | sed -n '330,370p'

## 144. tool

_create_time: 1783433689.7792096_

The output of this plugin was redacted.

## 145. assistant

_create_time: 1783433692.326145_

[empty message]

## 146. assistant

_create_time: 1783433689.778294_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/user_state_http.py | sed -n '590,670p' && printf '\n--- dispatch snippets ---\n' && nl -ba polylogue/daemon/user_state_http.py | sed -n '670,760p'

## 147. tool

_create_time: 1783433693.2718008_

The output of this plugin was redacted.

## 148. assistant

_create_time: 1783433693.270605_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/server.py | sed -n '40,90p;95,170p;176,230p;238,285p;285,380p;380,420p' && printf '\n--- receiver relevant ---\n' && nl -ba polylogue/browser_capture/receiver.py | sed -n '44,75p;100,120p;318,360p;478,612p'

## 149. tool

_create_time: 1783433696.9412398_

The output of this plugin was redacted.

## 150. assistant

_create_time: 1783433700.216114_

[empty message]

## 151. assistant

_create_time: 1783433696.940178_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/server.py | sed -n '259,379p'

## 152. tool

_create_time: 1783433700.1615357_

The output of this plugin was redacted.

## 153. assistant

_create_time: 1783433703.751876_

[empty message]

## 154. assistant

_create_time: 1783433700.150247_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/server.py | sed -n '379,420p'

## 155. tool

_create_time: 1783433702.8110864_

The output of this plugin was redacted.

## 156. assistant

_create_time: 1783433702.810043_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/config.py | sed -n '1018,1032p;1495,1545p' && nl -ba polylogue/daemon/cli.py | sed -n '960,985p;1548,1612p'

## 157. tool

_create_time: 1783433706.1134193_

The output of this plugin was redacted.

## 158. assistant

_create_time: 1783433711.512295_

[empty message]

## 159. assistant

_create_time: 1783433706.112433_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "api_auth_token|api_host|api_port|api_is_remote|remote.*api|daemon_api" polylogue/config.py polylogue/daemon/cli.py | sed -n '1,160p'

## 160. tool

_create_time: 1783433709.7639177_

The output of this plugin was redacted.

## 161. assistant

_create_time: 1783433709.762662_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/config.py | sed -n '1458,1496p' && nl -ba polylogue/daemon/cli.py | sed -n '858,874p;995,1005p'

## 162. tool

_create_time: 1783433714.545128_

The output of this plugin was redacted.

## 163. assistant

_create_time: 1783433718.858275_

[empty message]

## 164. assistant

_create_time: 1783433714.544051_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/healthz.py | sed -n '40,85p;150,175p' && nl -ba polylogue/daemon/metrics.py | sed -n '1830,1855p;1700,1805p'

## 165. tool

_create_time: 1783433718.150865_

The output of this plugin was redacted.

## 166. assistant

_create_time: 1783433724.036695_

[empty message]

## 167. assistant

_create_time: 1783433718.149898_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "polylogue_archive|polylogue_daemon|archive_.*bytes|db_size|status|uptime|source" polylogue/daemon/metrics.py | sed -n '1,120p'

## 168. tool

_create_time: 1783433723.2015233_

The output of this plugin was redacted.

## 169. assistant

_create_time: 1783433725.857437_

[empty message]

## 170. assistant

_create_time: 1783433723.200477_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba docs/security.md | sed -n '1,180p'

## 171. tool

_create_time: 1783433726.0644214_

The output of this plugin was redacted.

## 172. assistant

_create_time: 1783433729.415833_

[empty message]

## 173. assistant

_create_time: 1783433726.063398_

bash -lc cd /mnt/data/polylogue_work/polylogue && find .. -path '*003_polylogue_kwsb_1.md' -o -path '*prework*' | sed -n '1,80p' && rg -n "polylogue-kwsb\.1|Host/Origin|spool governor|Sec-Fetch|receiver token|access_token" -S . | sed -n '1,200p'

## 174. tool

_create_time: 1783433730.1315184_

The output of this plugin was redacted.

## 175. assistant

_create_time: 1783433735.171994_

[empty message]

## 176. assistant

_create_time: 1783433730.130354_

bash -lc find /mnt/data/polylogue_src -maxdepth 5 -type f | sed -n '1,200p' | grep -E '003_polylogue|beads|kwsb|prework' || true && find /mnt/data -maxdepth 5 -type f | grep -E '003_polylogue|beads|kwsb|prework' | sed -n '1,200p'

## 177. tool

_create_time: 1783433735.6871874_

The output of this plugin was redacted.

## 178. assistant

_create_time: 1783433740.278433_

[empty message]

## 179. assistant

_create_time: 1783433735.685973_

bash -lc cd /mnt/data/polylogue_src/polylogue && rg -n "polylogue-kwsb\.1|Daemon/capture security|Host/Origin|spool governor|Sec-Fetch|access_token|receiver" polylogue-beads.md polylogue-beads.xml polylogue-beads-export.jsonl | sed -n '1,240p'

## 180. tool

_create_time: 1783433739.9343772_

The output of this plugin was redacted.

## 181. assistant

_create_time: 1783433754.699788_

[empty message]

## 182. assistant

_create_time: 1783433739.933193_

bash -lc cd /mnt/data/polylogue_work/polylogue && rg -n "do_(PUT|PATCH|HEAD)|WebSocket|websocket|Upgrade|Connection: Upgrade|ThreadingHTTPServer|HTTPServer|serve_forever|add_route|route\(" polylogue/daemon polylogue/browser_capture browser-extension | sed -n '1,200p'

## 183. tool

_create_time: 1783433755.287838_

The output of this plugin was redacted.

## 184. assistant

_create_time: 1783433761.888918_

[empty message]

## 185. assistant

_create_time: 1783433755.286561_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/http.py | sed -n '3618,3645p' && nl -ba polylogue/daemon/browser_capture.py | sed -n '1,90p'

## 186. tool

_create_time: 1783433759.929641_

The output of this plugin was redacted.

## 187. assistant

_create_time: 1783433763.817185_

[empty message]

## 188. assistant

_create_time: 1783433759.928374_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/daemon/test_daemon_http_security.py | sed -n '1,240p' && printf '\n--- contracts ---\n' && nl -ba tests/unit/daemon/test_daemon_http_contracts.py | sed -n '1,140p' && printf '\n--- events query token test ---\n' && nl -ba tests/unit/daemon/test_daemon_events_endpoint.py | sed -n '350,375p'

## 189. tool

_create_time: 1783433763.7542922_

The output of this plugin was redacted.

## 190. assistant

_create_time: 1783433763.753064_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/daemon/test_daemon_http_security.py | sed -n '240,380p'

## 191. tool

_create_time: 1783433768.1644168_

The output of this plugin was redacted.

## 192. assistant

_create_time: 1783433768.101531_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/daemon/test_daemon_http_security.py | sed -n '380,460p'

## 193. tool

_create_time: 1783433770.896604_

The output of this plugin was redacted.

## 194. assistant

_create_time: 1783433774.531955_

[empty message]

## 195. assistant

_create_time: 1783433776.030194_

[empty message]

## 196. assistant

_create_time: 1783433770.890158_

bash -lc cd /mnt/data/polylogue_work/polylogue && find tests/unit/browser_capture -type f -maxdepth 2 -print -exec sed -n '1,120p' {} \; | sed -n '1,220p'

## 197. tool

_create_time: 1783433774.300432_

The output of this plugin was redacted.

## 198. assistant

_create_time: 1783433782.732386_

[empty message]

## 199. assistant

_create_time: 1783433774.299296_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/browser_capture/route_contracts.py | sed -n '1,220p' && rg -n "auth|origin|token|status|browser-captures|post-commands|archive-state" tests/unit/browser_capture/test_receiver.py | sed -n '1,180p'

## 200. tool

_create_time: 1783433780.5287719_

The output of this plugin was redacted.

## 201. assistant

_create_time: 1783433784.468759_

[empty message]

## 202. assistant

_create_time: 1783433780.527611_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/browser_capture/test_receiver.py | sed -n '120,180p;195,250p;440,505p;505,590p'

## 203. tool

_create_time: 1783433784.7565408_

The output of this plugin was redacted.

## 204. assistant

_create_time: 1783433788.967851_

[empty message]

## 205. assistant

_create_time: 1783433784.755203_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba tests/unit/browser_capture/test_receiver.py | sed -n '505,670p'

## 206. tool

_create_time: 1783433788.531427_

The output of this plugin was redacted.

## 207. assistant

_create_time: 1783433794.235824_

[empty message]

## 208. assistant

_create_time: 1783433788.529889_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/daemon/web_shell.py | sed -n '405,420p;1090,1125p;1280,1295p' && printf '\n--- reader escaping sinks ---\n' && nl -ba polylogue/daemon/web_shell_reader.py | sed -n '355,390p;430,445p;270,285p;205,225p' && printf '\n--- renderer sanitizer ---\n' && nl -ba polylogue/rendering/renderers/html_template.py | sed -n '15,35p' && nl -ba polylogue/rendering/templates/session.html | sed -n '320,350p' && nl -ba polylogue/rendering/renderers/html_sanitizer.py | sed -n '95,122p' && nl -ba polylogue/rendering/renderers/html_highlighting.py | sed -n '60,108p'

## 209. tool

_create_time: 1783433795.0086367_

The output of this plugin was redacted.

## 210. assistant

_create_time: 1783433795.007459_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/pipeline/ids.py | sed -n '1,80p' && rg -n "provider_session_id|legacy_browser_capture_native_id|session_id" polylogue/sources/parsers/browser_capture.py polylogue/browser_capture/identity.py polylogue/browser_capture/models.py | sed -n '1,120p'

## 211. tool

_create_time: 1783433799.1857605_

The output of this plugin was redacted.

## 212. assistant

_create_time: 1783433805.666637_

[empty message]

## 213. assistant

_create_time: 1783433799.18451_

bash -lc cd /mnt/data/polylogue_work/polylogue && nl -ba polylogue/sources/parsers/browser_capture.py | sed -n '130,210p' && nl -ba polylogue/browser_capture/models.py | sed -n '84,105p;154,174p'

## 214. tool

_create_time: 1783433803.4716055_

The output of this plugin was redacted.

## 215. assistant

_create_time: 1783433817.562361_

[empty message]

## 216. assistant

_create_time: 1783433831.399025_

[empty message]

## 217. assistant

_create_time: 1783433803.470358_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
import ast, re
from pathlib import Path
p=Path('polylogue/daemon/http.py')
s=p.read_text()
mod=ast.parse(s)
deflines={}
for node in ast.walk(mod):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
        deflines[node.name]=node.lineno
print('provider handler line',deflines.get('_handle_provider_usage'))
for name in ['_handle_sources','_handle_events','_handle_otlp_post','_handle_reset','_handle_ingest','_handle_maintenance_plan','_handle_maintenance_run']:
    print(name,deflines.get(name))

# user_state defs
up=Path('polylogue/daemon/user_state_http.py')
um=ast.parse(up.read_text())
uf={node.name:node.lineno for node in ast.walk(um) if isinstance(node,ast.FunctionDef)}
for name in ['handle_list_marks','handle_create_mark','handle_delete_mark']:
 print(name,uf.get(name))
PY

## 218. tool

_create_time: 1783433829.1188502_

The output of this plugin was redacted.

## 219. assistant

_create_time: 1783433871.018123_

[empty message]

## 220. assistant

_create_time: 1783433877.007013_

[empty message]

## 221. assistant

_create_time: 1783433877.562303_

[empty message]

## 222. assistant

_create_time: 1783433878.272986_

[empty message]

## 223. assistant

_create_time: 1783433829.117546_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
import ast,re
from pathlib import Path
p=Path('polylogue/daemon/http.py')
text=p.read_text().splitlines()
# simple extraction
for start,end,name in [(228,254,'static'),(257,276,'param'),(285,290,'otlp'),(321,327,'post')]:
    print('\n'+name)
    for i in range(start-1,end):
        line=text[i]
        m=re.search(r'"(/[^"]+)"',line)
        if m:
            print(i+1,m.group(1),line.strip())
PY

## 224. tool

_create_time: 1783433875.4791107_

The output of this plugin was redacted.

## 225. assistant

_create_time: 1783434025.074448_

[empty message]

## 226. assistant

_create_time: 1783433875.477752_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
from pathlib import Path

report_path = Path('/mnt/data/polylogue_daemon_capture_attack_surface_audit.md')

# Hand-maintained, source-anchored inventory from the audited working tree.
daemon_rows = [
    # method, path, handler, auth, host/origin, exposure, coverage
    ("GET", "/", "polylogue/daemon/http.py:_serve_web_shell L1397; dispatch L1191-L1202", "Unauth loopback bootstrap when api_host + client socket are loopback; otherwise bearer-if-configured.", "No HTTP Host check; no Origin/Sec-Fetch check.", "Web shell HTML bootstrap. The shell JS calls archive/user APIs.", "Covered only if kwsb.1 gate runs before shell-bootstrap special cases."),
    ("GET", "/s/:session_id", "polylogue/daemon/http.py:_serve_web_shell L1397; dispatch L1194-L1202", "Same as /.", "No Host/Origin.", "Session deep-link shell bootstrap.", "Same as /."),
    ("GET", "/w/:mode", "polylogue/daemon/http.py:_serve_web_shell L1397; dispatch L1194-L1202; modes in workspace_routes.WORKSPACE_SHELL_MODES", "Same as /.", "No Host/Origin.", "Workspace shell bootstrap for tabs/stack/compare/timeline.", "Same as /."),
    ("GET", "/p", "polylogue/daemon/http.py:_serve_paste_browser_page L1402; dispatch L1208-L1212", "Unauth loopback bootstrap; otherwise bearer-if-configured.", "No Host/Origin.", "Paste-browser standalone page.", "Covered only if pre-dispatch gate includes non-/api shell pages."),
    ("GET", "/a", "polylogue/daemon/http.py:_serve_attachment_library_page L1405; dispatch L1217-L1221", "Unauth loopback bootstrap; otherwise bearer-if-configured.", "No Host/Origin.", "Attachment-library standalone page.", "Covered only if pre-dispatch gate includes non-/api shell pages."),
    ("GET", "/healthz/live", "polylogue/daemon/healthz.py:handle_healthz_live L40; dispatch L1228-L1232", "None.", "No Host/Origin.", "Operational liveness; includes pid, started_at, uptime.", "Covered only if Host gate precedes unauth healthz exception."),
    ("GET", "/healthz/ready", "polylogue/daemon/healthz.py:handle_healthz_ready L65; dispatch L1233-L1237", "None.", "No Host/Origin.", "Readiness; checks/FTS/degraded details; exception message can be reflected in 503.", "Covered only if Host gate precedes unauth healthz exception."),
    ("GET", "/metrics", "polylogue/daemon/metrics.py:handle_metrics L1833; dispatch L1244-L1249", "None.", "No Host/Origin.", "Prometheus series: uptime/build/status, DB sizes, archive tiers, source/message counts, ingest/FTS/embedding/debt state; errors reflected as metric labels.", "Covered only if Host gate precedes unauth metrics exception; auth tightening is a separate policy choice."),
]

static = [
("/api/health/check","_handle_health_check L1583","bearer-if-configured; query access_token fallback accepted globally","No Host/Origin","Health check detail."),
("/api/health","_handle_health L1609","bearer-if-configured; global query-token fallback","No Host/Origin","Health/DB/disk summary."),
("/api/status","_handle_status L1653","bearer-if-configured; global query-token fallback","No Host/Origin","Daemon status, component state, watcher roots, DB sizes, failure samples, browser-capture state."),
("/api/dev-loop","_handle_dev_loop L1672","bearer-if-configured; global query-token fallback","No Host/Origin","Branch/dev-loop metadata: run id, log dir, archive root, ports, pid, cwd."),
("/api/events","_handle_events L1680 -> events_http.handle_events L75","bearer-if-configured; query access_token fallback used for EventSource and poll","No Host/Origin","SSE stream or JSON event poll; reads event ledger and payloads."),
("/api/agents/coordination","_handle_agent_coordination L1408","bearer-if-configured; global query-token fallback","No Host/Origin","Agent coordination envelope."),
("/api/sessions","_handle_list_sessions L1691","bearer-if-configured; global query-token fallback","No Host/Origin","Archive session list/search."),
("/api/facets","_handle_facets L3271","bearer-if-configured; global query-token fallback","No Host/Origin","Archive facets/aggregates."),
("/api/provider-usage","registered at _static_get_routes L238, but no _handle_provider_usage method exists in http.py","bearer-if-configured before dispatch","No Host/Origin","Intended provider-usage diagnostics; currently an implemented-pattern/contract drift that can raise AttributeError after auth.",),
("/api/query-units","_handle_query_units L2705","bearer-if-configured; global query-token fallback","No Host/Origin","Query unit/evidence payload."),
("/api/archive-debt","_handle_archive_debt L2763","bearer-if-configured; global query-token fallback","No Host/Origin","Archive debt rows and diagnostics."),
("/api/import/explain","_handle_import_explain L2784","bearer-if-configured; global query-token fallback","No Host/Origin","Import/source explanation; may include path evidence depending request."),
("/api/refs/resolve","_handle_ref_resolve L2818","bearer-if-configured; global query-token fallback","No Host/Origin","Public reference resolution."),
("/api/query-completions","_handle_query_completions L2840","bearer-if-configured; global query-token fallback","No Host/Origin","Query completion metadata."),
("/api/action-affordances","_handle_action_affordances L2857","bearer-if-configured; global query-token fallback","No Host/Origin","Action affordance inventory."),
("/api/read-view-profiles","_handle_read_view_profiles L2866","bearer-if-configured; global query-token fallback","No Host/Origin","Read-view profile metadata."),
("/api/assertions","_handle_assertions L2882","bearer-if-configured; global query-token fallback","No Host/Origin","Assertion overlay claims."),
("/api/paste-browser","_handle_paste_browser L1418","bearer-if-configured; global query-token fallback","No Host/Origin","Paste-browser API data."),
("/api/attachments","_handle_attachment_library L1484","bearer-if-configured; global query-token fallback","No Host/Origin","Attachment library API data."),
("/api/stack","_handle_stack L3189","bearer-if-configured; global query-token fallback","No Host/Origin","Workspace stack projection."),
("/api/compare","_handle_compare L3202","bearer-if-configured; global query-token fallback","No Host/Origin","Workspace compare projection."),
("/api/sources","_handle_sources L3358","bearer-if-configured; global query-token fallback","No Host/Origin","Source list; docs state absolute source roots are returned to authenticated callers."),
("/api/thread-continue-templates","_handle_get_thread_continue_templates L2693","bearer-if-configured; global query-token fallback","No Host/Origin","Thread-continuation URL templates."),
("/api/maintenance/operations","_handle_maintenance_operations L3613","bearer-if-configured; global query-token fallback","No Host/Origin","Maintenance operation inventory/status list."),
]
for p,h,a,o,e in static:
    cov = "Covered only if central gate covers all GET route-table entries; not covered by POST-only Origin hardening. Also restrict query-token fallback for this row unless this is /api/events."
    daemon_rows.append(("GET", p, f"polylogue/daemon/http.py:{h}", a, o, e, cov))

params = [
("/api/sessions/:id","_handle_get_session L2213","Session detail."),
("/api/sessions/:id/messages","_handle_get_messages L3111","Session messages."),
("/api/sessions/:id/read","_handle_get_session_read L2942","Session read-view envelope."),
("/api/sessions/:id/raw","_handle_get_session_raw L2398","Raw session payload JSON."),
("/api/sessions/:id/cost","_handle_get_session_cost L2432","Cost/usage for session."),
("/api/sessions/:id/provenance","_handle_get_session_provenance L2578","Provenance; raw bytes opt-in via include_raw."),
("/api/sessions/:id/topology","_handle_get_session_topology L2618","Session topology."),
("/api/sessions/:id/topology/parent-chain","_handle_get_session_parent_chain L2658","Parent-chain topology."),
("/api/sessions/:id/similar","_handle_get_session_similar L3075","Similar sessions."),
("/api/sessions/:id/attachments","_handle_get_session_attachments L1558","Session attachments."),
("/api/insights/sessions/:id","_handle_get_session_insights L2461","Session insight envelope."),
("/api/raw_artifacts/:id","_handle_get_raw_artifact L3255","Raw artifact preview."),
("/api/maintenance/status/:id","_handle_maintenance_status L3606","Maintenance operation status."),
]
for p,h,e in params:
    daemon_rows.append(("GET", p, f"polylogue/daemon/http.py:{h}; registered in _parameterized_get_routes L257-L275", "bearer-if-configured; global query-token fallback", "No Host/Origin", e, "Covered only if central gate covers parameterized GET routes; not covered by POST-only Origin hardening."))

post = [
("POST","/v1/traces","polylogue/daemon/http.py:_handle_otlp_post L3418; otlp_receiver.handle_traces L102","observability_enabled required; if api_host loopback, no bearer required; if non-loopback, bearer required","No Host; no Origin/Sec-Fetch; OTLP branch runs before normal auth/origin path","Writes raw telemetry payload into ops.db, bounded only by configured otlp_max_body_bytes.","Covered only if gate is before the OTLP special branch in _do_post_impl L1328-L1354."),
("POST","/v1/metrics","polylogue/daemon/http.py:_handle_otlp_post L3418; otlp_receiver.handle_metrics L109","same as /v1/traces","same as /v1/traces","Writes raw telemetry payload into ops.db.","same condition as /v1/traces"),
("POST","/v1/logs","polylogue/daemon/http.py:_handle_otlp_post L3418; otlp_receiver.handle_logs L118","same as /v1/traces","same as /v1/traces","Writes raw telemetry payload into ops.db.","same condition as /v1/traces"),
("POST","/api/reset","polylogue/daemon/http.py:_handle_reset L3372","bearer-if-configured; then Origin loopback if Origin present; absent Origin allowed","No Host; Origin absent allowed; no Sec-Fetch","Destructive archive reset.","Covered if gate precedes _authenticated_post_routes and distinguishes CLI no-Origin from browser cross-site."),
("POST","/api/ingest","polylogue/daemon/http.py:_handle_ingest L3452","same as /api/reset","same as /api/reset","Schedules/imports staged inbox artifact; writes daemon event.","same as /api/reset"),
("POST","/api/maintenance/plan","polylogue/daemon/http.py:_handle_maintenance_plan L3533","same as /api/reset","same as /api/reset","Maintenance preview/plan; may read archive state.","same as /api/reset"),
("POST","/api/maintenance/run","polylogue/daemon/http.py:_handle_maintenance_run L3569","same as /api/reset","same as /api/reset","Runs maintenance/backfill operations.","same as /api/reset"),
]
daemon_rows.extend(post)

user_get = [
("GET","/api/user/marks","handle_list_marks L179","Reads user.db marks."),
("GET","/api/user/annotations","handle_list_annotations L274","Reads annotations."),
("GET","/api/user/annotations/:id","handle_get_annotation L310","Reads one annotation."),
("GET","/api/user/saved-views","handle_list_saved_views L325","Reads saved views."),
("GET","/api/user/saved-views/:id","handle_get_saved_view L346","Reads one saved view."),
("GET","/api/user/recall-packs","handle_list_recall_packs L361","Reads recall packs."),
("GET","/api/user/recall-packs/:id","handle_get_recall_pack L381","Reads one recall pack."),
("GET","/api/user/workspaces","handle_list_workspaces L396","Reads workspaces."),
("GET","/api/user/workspaces/:id","handle_get_workspace L417","Reads one workspace."),
]
# some line numbers may be off? Let's verify from file? We can accept? Better check maybe line 274 etc. 
# Actually generated from source? Let's replace with known? We'll later maybe fine.
for m,p,h,e in user_get:
    daemon_rows.append((m,p,f"polylogue/daemon/user_state_http.py:{h}; registered in _routes L600-L610; dispatched by http.py L1263-L1265", "bearer-if-configured; global query-token fallback via daemon _check_auth", "No Host/Origin", e, "Covered if central gate applies before user_state dispatch."))

user_mut = [
("POST","/api/user/marks","handle_create_mark L202","Writes mark."),
("POST","/api/user/annotations","handle_save_annotation L330?","Writes annotation."),
("POST","/api/user/saved-views","handle_save_view L?","Writes saved view."),
("POST","/api/user/recall-packs","handle_save_recall_pack L?","Writes recall pack."),
("POST","/api/user/workspaces","handle_save_workspace L?","Writes workspace."),
("DELETE","/api/user/marks","handle_delete_mark L244","Deletes mark(s)."),
("DELETE","/api/user/annotations/:id","handle_delete_annotation L?","Deletes annotation."),
("DELETE","/api/user/saved-views/:id","handle_delete_saved_view L?","Deletes saved view."),
("DELETE","/api/user/recall-packs/:id","handle_delete_recall_pack L?","Deletes recall pack."),
("DELETE","/api/user/workspaces/:id","handle_delete_workspace L?","Deletes workspace."),
]
for m,p,h,e in user_mut:
    daemon_rows.append((m,p,f"polylogue/daemon/user_state_http.py:{h}; registered in _routes L611-L620; dispatched by http.py L1364-L1368 or L1390-L1392", "bearer-if-configured; then Origin loopback if Origin present; absent Origin allowed", "No Host; Origin absent allowed; no Sec-Fetch", e, "Covered if central gate applies before user_state POST/DELETE dispatch."))

daemon_rows.append(("OPTIONS","*","polylogue/daemon/http.py:do_OPTIONS L1182","None; always 405","No Host/Origin before 405","No CORS advertised.","Gate should reject bad Host before method-level 405; keep good-host 405."))

capture_rows = [
("OPTIONS","*","polylogue/browser_capture/server.py:_do_options L178","No bearer on preflight by design","Origin allowlist only; absent Origin allowed; no Host gate","CORS methods/headers disclosure for any allowed/absent Origin.","Covered only by a receiver-side gate; daemon gate does not touch this server."),
("GET","/v1/status","polylogue/browser_capture/server.py:_do_get L195-L201 -> receiver_status_payload L344","bearer only if configured; default config has auth_token None","Origin allowlist; absent Origin allowed; chrome-extension://* wildcard; no Host gate","Receiver status; exposes absolute spool_path, allowed_origins, allow_remote, auth_required.","Needs receiver token lifecycle + receiver Host/Origin admission."),
("GET","/v1/archive-state","polylogue/browser_capture/server.py:_do_get L202-L218 -> existing_capture_state L357","bearer only if configured; default none","Origin allowlist; absent Origin allowed; no Host gate","Reads spool/archive state for arbitrary provider/provider_session_id.","Needs receiver token + Host/Origin admission."),
("GET","/v1/post-commands","polylogue/browser_capture/server.py:_do_get L219-L236 -> poll_post_commands L552","bearer only if configured; default none","Origin allowlist; absent Origin allowed; no Host gate","Polls and *claims* pending post commands (pending -> dispatched) when posting is enabled; this is a mutating GET.","Needs receiver token + Host/Origin admission; treat as state-changing despite GET."),
("POST","/v1/browser-captures","polylogue/browser_capture/server.py:_do_post L259-L307 -> write_capture_envelope L318","bearer only if configured; default none","Origin allowlist; absent Origin allowed; chrome-extension wildcard; no Host gate","Writes capture JSON to spool. Per-request body cap 128 MiB, no aggregate quota.","Needs receiver token + Host/Origin admission + spool governor."),
("POST","/v1/post-commands","polylogue/browser_capture/server.py:_post_command_enqueue L309-L348 -> enqueue_post_command L520","bearer only if configured; default none; env POLYLOGUE_BROWSER_POST_ENABLED=1 also required","Origin allowlist; absent Origin allowed; no Host gate","Writes outbound post-command JSON queue; can instruct extension to fill/submit provider composer when polled.","Needs receiver token + Host/Origin admission + queue quota."),
("POST","/v1/post-commands/:id/ack","polylogue/browser_capture/server.py:_post_command_ack L350-L382 -> ack_post_command L587","bearer only if configured; default none","Origin allowlist; absent Origin allowed; no Host gate","Mutates queued post command status/detail/observed_url.","Needs receiver token + Host/Origin admission."),
]

# Fix exact user_state line numbers by parsing AST
import ast
us_path=Path('polylogue/daemon/user_state_http.py')
uf={}
for node in ast.walk(ast.parse(us_path.read_text())):
    if isinstance(node, ast.FunctionDef):
        uf[node.name]=node.lineno
# Replace '?' rows. Simple map.
# Not necessary in rows already but appendix will state exact.

# Helper to build markdown table
def md_table(headers, rows):
    out=[]
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('| ' + ' | '.join(['---']*len(headers)) + ' |')
    for row in rows:
        safe=[]
        for cell in row:
            safe.append(str(cell).replace('|','\\|').replace('\n','<br>'))
        out.append('| ' + ' | '.join(safe) + ' |')
    return '\n'.join(out)

summary_findings = [
("F1", "Central admission must be truly pre-dispatch.", "The unauth shell, healthz, metrics, and OTLP branches are all reached before the normal authenticated API route tables; a gate built only around _static_get_routes/_parameterized_get_routes or POST Origin logic leaves live holes."),
("F2", "SSE is not a separate protocol route, but it is a GET route with its own token leak mode.", "_check_auth accepts ?access_token= on every authenticated route, not just /api/events. do_GET client-disconnect debug logging records self.path, so /api/events?access_token=... can leak in debug logs."),
("F3", "OTLP is an overlooked write surface.", "When observability_enabled is true and api_host is loopback, /v1/traces,/metrics,/logs bypass bearer and never call _check_cross_origin. They store raw telemetry bytes in ops.db behind only the body-size cap."),
("F4", "Operational endpoints are unauthenticated and more revealing than their comments imply.", "/healthz/live includes pid/start time; /healthz/ready can include exception message; /metrics exposes archive layout, DB sizes, source/message counts, ingest/failure/freshness state. Even if intentionally unauthenticated, they need Host/Sec-Fetch admission."),
("F5", "The receiver is a separate server; daemon middleware cannot protect it.", "Loopback receiver default has auth_token None, normal == comparison, Origin absent allowed, chrome-extension wildcard allowed, no Host gate. The extension stores receiverAuthToken manually in chrome.storage.local; there is no auto-mint/0600/rotation/provisioning path in the audited source."),
("F6", "The spool governor has to cover more than capture JSON.", "write_capture_envelope has a 128 MiB per-request HTTP cap but no aggregate quota, unique-session count cap, temp-file cleanup semantics, burst/backpressure behavior, or queue quota for post-commands."),
("F7", "Hostile captured content is mostly escaped in message bodies, but non-body identifiers hit inline-JS sinks.", "The standalone renderer uses Jinja autoescape + nh3 sanitization and MarkdownIt with html disabled. The web shell client-side renderer escapes message text, but escAttr only handles quotes and is used inside onclick JavaScript string literals with content-derived session/message IDs and anchors."),
("F8", "Route-contract drift exists outside the known holes.", "/api/provider-usage is registered in _static_get_routes and route_contracts, but http.py has no _handle_provider_usage method, so the route can raise after auth rather than returning a contract payload."),
]

threat_rows = [
("Shell/static pages", "DNS rebinding can load the shell under an attacker Host because loopback bootstrap uses socket client/api host, not HTTP Host. Same-origin JS from the attacker origin can then call local GET APIs if no token is configured.", "Other UID local process can connect to loopback; if no API token it can read shell/API. Same UID can read DB/token files directly; not a defendable boundary with 0600 files.", "Shell templates insert content-derived IDs/anchors into inline handlers; message bodies are escaped, but ID/anchor injection remains plausible."),
("Archive read APIs", "All GET /api read routes are reachable by DNS rebinding when token is absent; lack of CORS does not matter because the browser sees attacker Host as same-origin after rebinding.", "Other UID local process can read routes when no token; with token, 0600 storage protects other UIDs only if file perms/parent dirs are right.", "Archive text is escaped on web rendering; raw/provenance endpoints intentionally expose raw payload to authenticated clients."),
("/api/events SSE/poll", "DNS-rebound page can open EventSource/read JSON poll if no token; with token, query-token fallback is overbroad and log-leakable on disconnect.", "Local process can poll event ledger; same UID out of scope; other UID needs bearer once configured.", "SSE payload is JSON encoded, but consumed by web shell; downstream renderer risks match archive read APIs."),
("User-state POST/DELETE", "Cross-origin POST with Origin is denied, but absent-Origin and Sec-Fetch are not considered. DNS rebinding with attacker Host still needs Host gate; modern browsers usually send Origin, but Fetch Metadata gives a cheaper browser signal.", "Local processes can mutate if no token; with token, other UID blocked only by token secrecy.", "User annotation/view/workspace names should be treated as captured/untrusted content and avoid inline handlers."),
("Maintenance/reset/ingest", "Same as other mutating daemon routes; /api/reset is destructive and /api/maintenance/run can rewrite/backfill archive state.", "Local process can trigger destructive operations when no token.", "Ingested content later reaches archive/web shell; no direct HTML reflection in the POST response noted."),
("OTLP", "If observability is enabled, DNS-rebound same-origin JS can POST telemetry to /v1/* because the OTLP branch skips normal Origin checks on loopback; raw payload is stored in ops.db.", "Any local process can write telemetry when observability is enabled and api_host is loopback, even if no API token; other UID included.", "Telemetry payload may later be inspected by tools; keep raw payload out of HTML without sanitizer."),
("Healthz/metrics", "DNS rebinding reads operational fingerprinting info. No archive text, but enough for path/layout/freshness/activity profiling.", "Local process reads regardless of token; intended for probes but still cross-UID visible on loopback.", "Readiness exception messages and metrics error labels reflect exception strings; keep them path/secret-scrubbed."),
("Browser-capture status/archive-state", "Normal web origins are blocked by Origin allowlist, but absent-Origin local requests and any allowed extension origin are accepted when no token. Host is unchecked.", "Any local process, including other UID, can query status/archive-state when token absent; same UID can read spool directly.", "Provider/session IDs from capture flow can become archive IDs and then web-shell identifiers."),
("Browser-capture ingest/post commands", "Web origins are Origin-blocked by default, but no receiver token means local processes can forge captures; if posting is enabled, unauth local processes can enqueue commands for the extension to claim.", "Other UID local process can write/ack/poll if no token. Same UID cannot be strongly isolated; token files in 0600 only separate other UIDs.", "Forged captures can plant untrusted IDs/text/attachments into spool/archive; text is escaped, identifiers need safer rendering."),
]

fix_gap_rows = [
("WebSocket/SSE bypass", "No WebSocket route exists. SSE is /api/events GET. It bypasses POST-only Origin today and needs the same Host admission as all GET routes. Query-token fallback must be restricted to /api/events and ideally to EventSource-compatible requests."),
("Sec-Fetch-Site", "Use as a complementary browser signal, not a replacement for Host. Reject Sec-Fetch-Site: cross-site on all daemon and receiver routes unless an explicit dev CORS origin is configured. It catches browser requests with missing Origin; DNS rebinding still requires Host allowlist."),
("Receiver token lifecycle", "Not implemented: config default is None; loopback receiver accepts without token; _check_token uses ==; extension reads/writes receiverAuthToken in chrome.storage.local manually; no auto-minted 0600 token file, rotation, dev-loop propagation, or non-disclosure test."),
("Spool governor semantics", "Not implemented. Need max bytes + max files + temp-file accounting + queue quota. Define legitimate-burst behavior: accept within quota, replace same artifact without charging twice, return 429 or 507 before writing when over quota, and emit an operator-visible event. Decide whether to drop newest, drop oldest, or backpressure; current code always writes."),
("Metrics/healthz disclosure", "Known packet mentions archive reads/capture/spool; operational endpoints also need Host admission. Consider token/opt-in for /metrics beyond Host gate if multi-user local machines matter."),
("Request reflection", "Daemon disconnect logs include self.path; redact query strings before logging. Healthz/metrics reflect exception strings; keep scrubber tests. Web shell reflects IDs/anchors into inline JS; replace inline handlers with data attributes + addEventListener or encode JS args with JSON.stringify."),
("Route drift", "Add a route-table callable test. /api/provider-usage currently registers to a missing method."),
("Receiver mutating GET", "GET /v1/post-commands claims commands. Admission must not infer read-only from GET; apply state-changing policy to this route."),
]

test_rows = [
("daemon-host-foreign-all-get", "Parametrize over ROUTE_CONTRACTS GET plus /,/p,/a,/healthz/*,/metrics", "_make_handler(method='GET', path=sample, host='evil.example:8766', client=127.0.0.1, auth='Bearer secret' where needed)", "DNS-rebinding Host", "403 host_denied; handler mocks not called."),
("daemon-host-foreign-post-delete", "All daemon POST/DELETE routes including /v1/traces,/v1/metrics,/v1/logs", "Good token where relevant, Origin absent, Host evil.example", "DNS-rebinding write", "403 before auth/origin/OTLP handler; observability_enabled true fixture for OTLP."),
("daemon-host-keep-shell", "/, /s/some-id, /w/stack, /p, /a", "Host 127.0.0.1:8766 or localhost:8766, client loopback", "Web shell own-origin keep-working", "200/handler called; no 403."),
("daemon-host-keep-probes", "/healthz/live, /healthz/ready, /metrics", "Host 127.0.0.1:8766", "System probe keep-working", "Existing response status/shape preserved."),
("daemon-query-token-only-events", "/api/events?poll=1&access_token=secret and /api/sessions?access_token=secret", "auth_token='secret'; no Authorization header", "Query token overbreadth", "/api/events accepted; /api/sessions returns 401."),
("daemon-query-token-log-redaction", "/api/events?access_token=secret", "Patch _dispatch_get to raise BrokenPipeError; capture logger.debug kwargs", "Token-in-URL leak", "Logged path contains no access_token and no raw query token."),
("daemon-sec-fetch-cross-site-post", "/api/reset and every bearer_and_same_origin route", "Authorization valid, Origin absent, Sec-Fetch-Site: cross-site", "Browser request with missing Origin", "403 fetch_metadata_denied."),
("daemon-sec-fetch-curl-keep", "/api/ingest or /api/user/marks", "Authorization valid, no Origin, no Sec-Fetch-Site", "CLI/hook keep-working", "Not 401/403; route mock invoked."),
("daemon-otlp-disabled-still-404", "/v1/traces", "Good Host, observability_enabled false", "Keep existing contract", "404 not_found."),
("daemon-otlp-loopback-bad-host", "/v1/traces", "observability_enabled true, api_host loopback, Host evil.example", "OTLP DNS rebinding write", "403; handle_traces not called."),
("daemon-route-callable", "implemented_daemon_route_patterns()", "For each static route, assert getattr(handler, handler_name) exists; sample-dispatch with handler mocks", "Route-contract drift", "/api/provider-usage fixed; no AttributeError route."),
("receiver-default-auth-required", "make_server(...) default", "Monkeypatch token-dir to tmp_path; create config/server no explicit token", "Unauth loopback receiver", "config.auth_token is set; token file mode 0600; status reports auth_required true and does not reveal token."),
("receiver-token-forged", "POST /v1/browser-captures", "In-process BrowserCaptureHandler fixture; Origin chrome-extension://polylogue-test; Authorization Bearer wrong", "Forged receiver token", "401/403; no spool files."),
("receiver-token-missing", "POST /v1/browser-captures", "Allowed extension Origin, no Authorization", "Unauth capture POST", "401; no spool files."),
("receiver-host-foreign", "GET/POST receiver routes", "Correct token + allowed Origin but Host evil.example:8765", "DNS rebinding / Host confusion", "403 before status/read/write/claim."),
("receiver-origin-cross-site", "POST /v1/browser-captures", "Correct token, Origin https://evil.example", "Hostile web origin", "403 origin_not_allowed; no spool files."),
("receiver-options-keep", "OPTIONS /v1/browser-captures", "Allowed extension/web origin, Access-Control-Request-Headers authorization, good Host", "Extension/web preflight keep-working", "204 with expected allow headers; no token required on preflight."),
("receiver-post-commands-get-is-mutating", "GET /v1/post-commands", "Seed pending command file; then bad Host/wrong token", "Claim bypass", "403/401 and command remains pending."),
("receiver-spool-quota-reject", "POST /v1/browser-captures", "Quota fixture with existing files near limit; valid token/origin; large valid capture", "Disk-fill", "507 or 429 before write; no temp orphan; emitted quota signal."),
("receiver-spool-replace-keep", "POST /v1/browser-captures same provider_session_id", "Quota already contains same artifact; replacement within configured semantics", "Legitimate recapture", "Accepted if replacement does not exceed quota; reports replaced true."),
("receiver-post-command-quota", "POST /v1/post-commands", "POLYLOGUE_BROWSER_POST_ENABLED=1; queue near quota", "Queue disk-fill/control-plane flood", "429/507; no new command file."),
("web-shell-id-injection", "web shell/sidebar/message action renderer", "Use session/message id `x\\');window.__pwned=1;//` and malicious anchor; render HTML string without browser", "Captured-content XSS", "Generated HTML contains no inline executable break-out; preferred assertion: no inline onclick uses untrusted args."),
("standalone-renderer-sanitizer-keep", "rendering/renderers HTML", "Text containing <script>, onclick, javascript: URI", "Captured HTML injection", "Sanitized output removes script/event/javascript while preserving safe text/links."),
]

content = []
content.append("# Polylogue daemon + browser-capture attack-surface audit\n")
content.append("Scope: uploaded Polylogue working tree under `/mnt/data/polylogue_work/polylogue`; listening surfaces are `polylogue/daemon/http.py` and `polylogue/browser_capture/server.py`/`receiver.py`. I enumerated daemon routes from `_static_get_routes`, `_parameterized_get_routes`, `_observability_post_routes`, `_authenticated_post_routes`, `implemented_daemon_route_patterns`, and `user_state_http._routes`; receiver routes from `BrowserCaptureHandler._do_get`, `_do_post`, and `_do_options`. I found no daemon/receiver WebSocket handler; SSE is `GET /api/events`.\n")
content.append("## Executive findings\n")
content.append(md_table(["ID","Finding","Audit result"], summary_findings))
content.append("\n\n## Complete route inventory\n")
content.append("The `auth today` column reports the implemented gate, not the intended contract. `query access_token fallback` means `_check_auth` rewrites any `?access_token=` into a bearer header before route dispatch.\n")
content.append("### Daemon HTTP routes\n")
content.append(md_table(["Method","Path/pattern","Handler source","Auth gate today","Host/Origin today","Exposes or mutates","kwsb.1 coverage condition"], daemon_rows))
content.append("\n\n### Browser-capture receiver routes\n")
content.append(md_table(["Method","Path/pattern","Handler source","Auth gate today","Host/Origin today","Exposes or mutates","kwsb.1 coverage condition"], capture_rows))
content.append("\n\n## Threat pass by route class\n")
content.append(md_table(["Route class","Hostile web page","Hostile local process","Hostile captured content"], threat_rows))
content.append("\n\n## Gaps in the planned kwsb.1 fix\n")
content.append(md_table(["Candidate/gap","Conclusion"], fix_gap_rows))
content.append("\n\n## Negative-test matrix\n")
content.append("The cases below fit the existing in-process handler style from `tests/unit/daemon/test_daemon_http_security.py` and `test_daemon_http_contracts.py`: instantiate handlers with fake server/client/header state, patch route handlers, and assert refusal/keep-working without a live socket. Existing browser-capture tests use live `HTTPConnection`; these cases should add a parallel in-process receiver-handler fixture so Host headers and mutation side effects can be asserted without binding a port.\n")
content.append(md_table(["Test id","Route(s)","Fixture setup","Attack/contract","Expected result"], test_rows))
content.append("\n\n## Implementation notes for the fix packet\n")
content.append("A safe admission shape is: parse and normalize Host first; reject non-loopback/non-configured Host before any special route; then apply Fetch Metadata and Origin policy according to route class; then auth; then route dispatch. For daemon mutating routes, allow no-Origin/no-Sec-Fetch only for bearer-authenticated non-browser clients so CLI/hooks keep working. For the receiver, default to a minted token and require bearer on every data route; OPTIONS can remain tokenless after Host+Origin admission.\n\n")
content.append("Restrict `?access_token=` to `/api/events`; scrub query strings from every log path; consider `Referrer-Policy: no-referrer` and `Cache-Control: no-store` on shell/SSE responses. The overbroad fallback is especially risky because tests already bless it for EventSource, but the implementation applies it globally.\n\n")
content.append("The receiver token lifecycle must be a full path, not only a comparison change: generate a high-entropy token file under an operator-private directory, create parent dirs with restrictive perms, write token file mode 0600, pass the token into the daemon-run receiver, expose a deliberate operator/extension provisioning channel, support rotation, and test that status/debug logs never include the token. The extension currently reads `receiverAuthToken` and `receiverBaseUrl` from `chrome.storage.local` and sends Authorization only when a token has been manually configured.\n\n")
content.append("For rendering, the durable fix is to stop concatenating untrusted values into inline event handlers. Emit data attributes using real HTML escaping, install event listeners via JS, and serialize dynamic JS strings with `JSON.stringify` if an inline context remains. The standalone renderer path is substantially stronger than the web shell path because it uses Jinja autoescape, MarkdownIt with HTML disabled, and nh3 sanitization.\n")

report_path.write_text('\n'.join(content), encoding='utf-8')
print(report_path)
print('rows', len(daemon_rows), len(capture_rows), len(test_rows))
PY

## 227. tool

_create_time: 1783434026.679658_

The output of this plugin was redacted.

## 228. assistant

_create_time: 1783434037.577894_

[empty message]

## 229. tool

_create_time: 1783434037.4165845_

The output of this plugin was redacted.

## 230. assistant

_create_time: 1783434038.891088_

I’ve narrowed scope to two HTTP listeners: the daemon server and separate browser-capture server. There are no WebSockets; server push is SSE at `GET /api/events`.

## 231. assistant

_create_time: 1783434026.675665_

bash -lc cd /mnt/data/polylogue_work/polylogue && python - <<'PY'
import ast
from pathlib import Path
uf={}
for node in ast.walk(ast.parse(Path('polylogue/daemon/user_state_http.py').read_text())):
 if isinstance(node, ast.FunctionDef): uf[node.name]=node.lineno
for name in ['handle_list_marks','handle_create_mark','handle_delete_mark','handle_list_annotations','handle_get_annotation','handle_save_annotation','handle_delete_annotation','handle_list_saved_views','handle_get_saved_view','handle_save_view','handle_delete_saved_view','handle_list_recall_packs','handle_get_recall_pack','handle_save_recall_pack','handle_delete_recall_pack','handle_list_workspaces','handle_get_workspace','handle_save_workspace','handle_delete_workspace']:
 print(f'{name} L{uf.get(name)}')
PY

## 232. assistant

_create_time: 1783434049.428834_

[empty message]

## 233. tool

_create_time: 1783434049.2823455_

The output of this plugin was redacted.

## 234. assistant

_create_time: 1783434054.740115_

[empty message]

## 235. assistant

_create_time: 1783434037.415242_

bash -lc python - <<'PY'
from pathlib import Path
p=Path('/mnt/data/polylogue_daemon_capture_attack_surface_audit.md')
s=p.read_text()
repls={
'handle_list_annotations L274':'handle_list_annotations L278',
'handle_get_annotation L310':'handle_get_annotation L299',
'handle_list_saved_views L325':'handle_list_saved_views L376',
'handle_get_saved_view L346':'handle_get_saved_view L385',
'handle_list_recall_packs L361':'handle_list_recall_packs L445',
'handle_get_recall_pack L381':'handle_get_recall_pack L454',
'handle_list_workspaces L396':'handle_list_workspaces L516',
'handle_get_workspace L417':'handle_get_workspace L525',
'handle_save_annotation L330?':'handle_save_annotation L310',
'handle_save_view L?':'handle_save_view L396',
'handle_save_recall_pack L?':'handle_save_recall_pack L465',
'handle_save_workspace L?':'handle_save_workspace L536',
'handle_delete_annotation L?':'handle_delete_annotation L359',
'handle_delete_saved_view L?':'handle_delete_saved_view L428',
'handle_delete_recall_pack L?':'handle_delete_recall_pack L499',
'handle_delete_workspace L?':'handle_delete_workspace L583',
}
for a,b in repls.items():
    s=s.replace(a,b)
p.write_text(s)
print('patched')
PY
rg -n "\?|L\?" /mnt/data/polylogue_daemon_capture_attack_surface_audit.md | head -20

## 236. assistant

_create_time: 1783434049.281068_

bash -lc sed -n '1,80p' /mnt/data/polylogue_daemon_capture_attack_surface_audit.md && sed -n '80,150p' /mnt/data/polylogue_daemon_capture_attack_surface_audit.md && sed -n '150,220p' /mnt/data/polylogue_daemon_capture_attack_surface_audit.md

## 237. tool

_create_time: 1783434052.8479507_

The output of this plugin was redacted.

## 238. assistant

_create_time: 1783434066.206472_

[empty message]

## 239. assistant

_create_time: 1783434052.846314_

bash -lc python - <<'PY'
from pathlib import Path
p=Path('/mnt/data/polylogue_daemon_capture_attack_surface_audit.md')
lines=p.read_text().splitlines()
seen=False; out=[]
for line in lines:
    if line.startswith('| daemon-host-foreign-post-delete |'):
        if seen:
            continue
        seen=True
    out.append(line)
p.write_text('\n'.join(out)+'\n')
print('duplicates', sum(1 for l in out if l.startswith('| daemon-host-foreign-post-delete |')))
PY

## 240. tool

_create_time: 1783434067.206303_

The output of this plugin was redacted.

## 241. assistant

_create_time: 1783434545.073106_

[empty message]

## 242. assistant

_create_time: 1783434545.073106_

[empty message]

## 243. assistant

_create_time: 1783434545.073106_

I treated the uploaded brief as the scope: verify the kwsb.1 Host/Origin/token/spool fix against the full daemon and browser-capture listening surface, not just the three confirmed holes. fileciteturn0file0

Bottom line: the planned fix is directionally right, but incomplete as stated. It needs to cover the daemon before every dispatch branch, the separate browser-capture server, the OTLP fast-path, unauthenticated health/metrics, global `?access_token=` behavior, and stored-content rendering sinks. I also found one registered daemon route that is currently broken: `GET /api/provider-usage` is in `_static_get_routes()` and route contracts, but `_handle_provider_usage` does not exist in `polylogue/daemon/http.py`.

I could not find the referenced `prework-v2/task_packets/task_packets/003_polylogue_kwsb_1.md` in the uploaded source bundle, so the “planned fix” comparison below is against the live bead design text in `polylogue-beads-export.jsonl` plus source anchors.

### Enumeration method

I enumerated daemon routes from these sources:

`polylogue/daemon/http.py:228` `_static_get_routes()`, `http.py:257` `_parameterized_get_routes()`, `http.py:285` `_observability_post_routes()`, `http.py:321` `_authenticated_post_routes()`, fixed dispatch branches in `http.py:1191-1249`, and dynamic user-state routes from `polylogue/daemon/user_state_http.py:600`.

I enumerated browser-capture receiver routes from branch dispatch in `polylogue/browser_capture/server.py:175-236` and `server.py:238-308`, cross-checking `polylogue/browser_capture/route_contracts.py:38`.

### Route inventory

| Surface | Method + path | Handler anchor | Auth today | Host / Origin today | Exposes or mutates | Planned gate coverage |
|---|---|---:|---|---|---|---|
| Web shell | `GET /` | `http.py:1193`, `http.py:1397` | Unauthenticated if bind host and client socket are loopback; otherwise bearer if configured | No Host check. No Origin check. | Main HTML/JS shell; then calls API. | Must be Host-gated before shell bootstrap; allow own loopback Host. |
| Web shell deep link | `GET /s/:session_id` | `http.py:1193`, `http.py:1397` | Same | No Host / no Origin | HTML shell seeded by URL path. | Covered only if central gate runs before this branch. |
| Workspace shell | `GET /w/:mode` for `tabs`, `stack`, `compare`, `timeline` | `http.py:1193`, `workspace_routes.py:17`, `http.py:1397` | Same | No Host / no Origin | HTML workspace shell. | Covered only if central gate runs before this branch. |
| Paste browser page | `GET /p` | `http.py:1208`, `http.py:1402` | Same | No Host / no Origin | Standalone HTML/JS paste browser. | Must be Host-gated. |
| Attachment page | `GET /a` | `http.py:1217`, `http.py:1405` | Same | No Host / no Origin | Standalone HTML/JS attachment library. | Must be Host-gated. |
| Liveness | `GET /healthz/live` | `http.py:1228`, `healthz.py:40` | None | No Host / no Origin | PID, start time, uptime. | Needs Host gate even if no bearer. |
| Readiness | `GET /healthz/ready` | `http.py:1233`, `healthz.py:65` | None | No Host / no Origin | Schema/FTS/check status; on exception returns `str(exc)` at `healthz.py:170`. | Needs Host gate; consider redacting exception text. |
| Metrics | `GET /metrics` | `http.py:1244`, `metrics.py:1833` | None | No Host / no Origin | Prometheus metrics: counts, DB/WAL sizes, ingest, embeddings, archive layout; errors include exception label at `metrics.py:1847`. | Needs Host gate; optional auth not required for local scraper. |
| Health check API | `GET /api/health/check` | `http.py:228`, `http.py:1583` | Bearer if configured; open if no token | No Host; no Origin | Health summary. | Covered if gate runs before static GET table. |
| Health API | `GET /api/health` | `http.py:231`, `http.py:1609` | Same | No Host; no Origin | DB/WAL sizes, disk free, quick DB check. | Covered. |
| Status API | `GET /api/status` | `http.py:232`, `http.py:1653` | Same | No Host; no Origin | Full daemon status, watcher roots, browser-capture state, readiness, failures. | Covered. |
| Dev-loop | `GET /api/dev-loop` | `http.py:233`, `http.py:1672`, payload at `http.py:303` | Same | No Host; no Origin | `run_id`, `log_dir`, `archive_root`, API/receiver ports, PID, CWD. | Covered; this route is more sensitive than “debug only.” |
| Events | `GET /api/events` | `http.py:234`, `http.py:1680`, `events_http.py:75` | Bearer if configured; also accepts `?access_token=` via global `_check_auth()` | No Host; no Origin | JSON poll or SSE event stream. | Covered only if Host gate includes SSE. Query-token must be restricted to this route. |
| Agent coordination | `GET /api/agents/coordination` | `http.py:235`, `http.py:1408` | Same | No Host; no Origin | Coordination envelope. | Covered. |
| Session list/search | `GET /api/sessions` | `http.py:236`, `http.py:1691` | Same | No Host; no Origin | Archive session summaries, titles, origins, tags, summaries, counts. | Covered. |
| Facets | `GET /api/facets` | `http.py:237`, `http.py:3271` | Same | No Host; no Origin | Query facets over archive metadata. | Covered. |
| Provider usage | `GET /api/provider-usage` | Registered at `http.py:238`; missing handler | Same before crash | No Host; no Origin | Intended usage diagnostics; currently unresolved handler causes server-side exception after auth. | Gate would cover admission, but route implementation is broken and needs a route-contract test. |
| Query units | `GET /api/query-units` | `http.py:239`, `http.py:2705` | Same | No Host; no Origin | Query-unit envelope over archive. | Covered. |
| Archive debt | `GET /api/archive-debt` | `http.py:240`, `http.py:2763` | Same | No Host; no Origin | Archive debt diagnostics. | Covered. |
| Import explain | `GET /api/import/explain` | `http.py:241`, `http.py:2784` | Same | No Host; no Origin | Import/source evidence; can redact or reveal paths via `no_redact`. | Covered. |
| Ref resolve | `GET /api/refs/resolve` | `http.py:242`, `http.py:2818` | Same | No Host; no Origin | Public object/evidence ref resolution. | Covered. |
| Query completions | `GET /api/query-completions` | `http.py:243`, `http.py:2840` | Same | No Host; no Origin | Query field/value metadata. | Covered. |
| Action affordances | `GET /api/action-affordances` | `http.py:244`, `http.py:2857` | Same | No Host; no Origin | Query/action metadata. | Covered. |
| Read-view profiles | `GET /api/read-view-profiles` | `http.py:245`, `http.py:2866` | Same | No Host; no Origin | Reader profile metadata. | Covered. |
| Assertions | `GET /api/assertions` | `http.py:246`, `http.py:2882` | Same | No Host; no Origin | User/agent overlay claims. | Covered. |
| Paste browser data | `GET /api/paste-browser` | `http.py:247`, `http.py:1418` | Same | No Host; no Origin | Captured paste evidence. | Covered. |
| Attachment library data | `GET /api/attachments` | `http.py:248`, `http.py:1484` | Same | No Host; no Origin | Attachment metadata, snippets, refs. | Covered. |
| Stack workspace data | `GET /api/stack` | `http.py:249`, `http.py:3189` | Same | No Host; no Origin | Multiple session payloads. | Covered. |
| Compare workspace data | `GET /api/compare` | `http.py:250`, `http.py:3202` | Same | No Host; no Origin | Pairwise session compare payload. | Covered. |
| Sources | `GET /api/sources` | `http.py:251`, `http.py:3358` | Same | No Host; no Origin | Configured watcher source names, roots, existence. | Covered. |
| Thread templates | `GET /api/thread-continue-templates` | `http.py:252`, `http.py:2693` | Same | No Host; no Origin | Continuation template metadata. | Covered. |
| Maintenance operations | `GET /api/maintenance/operations` | `http.py:253`, `http.py:3613` | Same | No Host; no Origin | Maintenance operation catalog. | Covered. |
| Session detail | `GET /api/sessions/:id` | `http.py:259`, `http.py:2213` | Same | No Host; no Origin | Session title, messages, text, attachments, tags, metadata. | Covered. |
| Session messages | `GET /api/sessions/:id/messages` | `http.py:260`, `http.py:3111` | Same | No Host; no Origin | Message page for a session. | Covered. |
| Session read view | `GET /api/sessions/:id/read` | `http.py:261`, `http.py:2942` | Same | No Host; no Origin | Profiled session read envelope. | Covered. |
| Session raw | `GET /api/sessions/:id/raw` | `http.py:262`, `http.py:2398` | Same | No Host; no Origin | Raw-artifact references and raw metadata. | Covered. |
| Session cost | `GET /api/sessions/:id/cost` | `http.py:263`, `http.py:2432` | Same | No Host; no Origin | Cost/usage panel. | Covered. |
| Session provenance | `GET /api/sessions/:id/provenance` | `http.py:264`, `http.py:2578` | Same | No Host; no Origin | Provenance; raw bytes available when requested. | Covered. |
| Session topology | `GET /api/sessions/:id/topology` | `http.py:265`, `http.py:2618` | Same | No Host; no Origin | Parent/child topology. | Covered. |
| Parent chain | `GET /api/sessions/:id/topology/parent-chain` | `http.py:266`, `http.py:2658` | Same | No Host; no Origin | Parent chain topology. | Covered. |
| Similar sessions | `GET /api/sessions/:id/similar` | `http.py:271`, `http.py:3075` | Same | No Host; no Origin | Similar-session search. | Covered. |
| Session attachments | `GET /api/sessions/:id/attachments` | `http.py:272`, `http.py:1558` | Same | No Host; no Origin | Attachments for one session. | Covered. |
| Session insights | `GET /api/insights/sessions/:id` | `http.py:273`, `http.py:2461` | Same | No Host; no Origin | Profile/timeline/phases/threads insights. | Covered. |
| Raw artifacts | `GET /api/raw_artifacts/:id` | `http.py:274`, `http.py:3255` | Same | No Host; no Origin | Raw artifact preview. | Covered. |
| Maintenance status | `GET /api/maintenance/status/:id` | `http.py:275`, `http.py:3606` | Same | No Host; no Origin | Maintenance run status. | Covered. |
| OTLP traces | `POST /v1/traces` | `http.py:285`, `http.py:3418` | If observability disabled: 404. If enabled and API bind is loopback: no bearer required. If non-loopback: bearer required. | No Host. No Origin. Routed before normal auth/origin at `http.py:1331`. | Writes raw telemetry to `ops.db`. Body capped by `otlp_max_body_bytes`. | Not covered by a POST-only app gate unless gate executes before OTLP special-case. |
| OTLP metrics | `POST /v1/metrics` | `http.py:288`, `http.py:3418` | Same | Same | Writes raw telemetry to `ops.db`. | Same gap. |
| OTLP logs | `POST /v1/logs` | `http.py:289`, `http.py:3418` | Same | Same | Writes raw telemetry to `ops.db`. | Same gap. |
| Reset | `POST /api/reset` | `http.py:323`, `http.py:3372` | Bearer if configured; open if no token | Origin checked only after auth and only if header present. No Host. | Deletes/reset state by scope. | Covered if gate handles absent-Origin browser cases and Host. |
| Ingest | `POST /api/ingest` | `http.py:324`, `http.py:3452` | Same | Same | Schedules staged inbox artifact. | Covered. |
| Maintenance plan | `POST /api/maintenance/plan` | `http.py:325`, `http.py:3533` | Same | Same | Creates maintenance preview/plan. | Covered. |
| Maintenance run | `POST /api/maintenance/run` | `http.py:326`, `http.py:3569` | Same | Same | Runs maintenance operation. | Covered. |
| User marks read | `GET /api/user/marks` | `user_state_http.py:602`, handler `:179` | Bearer if configured; open if no token | No Host; no Origin | Reads user marks. | Covered if dynamic dispatch included. |
| User annotations read | `GET /api/user/annotations`, `GET /api/user/annotations/:id` | `user_state_http.py:603-604`, handlers `:278`, `:299` | Same | No Host; no Origin | Reads notes/annotations. | Covered. |
| Saved views read | `GET /api/user/saved-views`, `GET /api/user/saved-views/:id` | `user_state_http.py:605-606`, handlers `:376`, `:385` | Same | No Host; no Origin | Reads saved query/view state. | Covered. |
| Recall packs read | `GET /api/user/recall-packs`, `GET /api/user/recall-packs/:id` | `user_state_http.py:607-608`, handlers `:445`, `:454` | Same | No Host; no Origin | Reads recall-pack state. | Covered. |
| Workspaces read | `GET /api/user/workspaces`, `GET /api/user/workspaces/:id` | `user_state_http.py:609-610`, handlers `:516`, `:525` | Same | No Host; no Origin | Reads workspace layouts. | Covered. |
| User marks mutate | `POST /api/user/marks`, `DELETE /api/user/marks` | `user_state_http.py:611`, `:616`; handlers `:202`, `:244` | Bearer if configured; open if no token | Origin checked only for POST/DELETE, skipped when absent. No Host. | Writes/deletes marks in user state DB. | Covered if dynamic routes included. |
| User annotations mutate | `POST /api/user/annotations`, `DELETE /api/user/annotations/:id` | `user_state_http.py:612`, `:617`; handlers `:310`, `:359` | Same | Same | Writes/deletes annotations. | Covered. |
| Saved views mutate | `POST /api/user/saved-views`, `DELETE /api/user/saved-views/:id` | `user_state_http.py:613`, `:618`; handlers `:396`, `:428` | Same | Same | Writes/deletes saved views. | Covered. |
| Recall packs mutate | `POST /api/user/recall-packs`, `DELETE /api/user/recall-packs/:id` | `user_state_http.py:614`, `:619`; handlers `:465`, `:499` | Same | Same | Writes/deletes recall packs. | Covered. |
| Workspaces mutate | `POST /api/user/workspaces`, `DELETE /api/user/workspaces/:id` | `user_state_http.py:615`, `:620`; handlers `:536`, `:583` | Same | Same | Writes/deletes workspace layouts. | Covered. |
| Daemon CORS preflight | `OPTIONS *` | `http.py:1184` | None | No Host; no Origin | Always 405. | Still should pass through Host admission or remain closed. |
| Receiver preflight | `OPTIONS *` | `browser_capture/server.py:175` | No bearer on preflight | Origin allowlist only; missing Origin allowed and emits `Access-Control-Allow-Origin: null`. No Host gate. | CORS preflight metadata. | Not covered by daemon gate; receiver needs its own admission. |
| Receiver status | `GET /v1/status` | `server.py:199`, `receiver.py:344` | Bearer only if configured; default no token | Origin allowlist; missing Origin allowed. No Host gate. | Receiver config status, including absolute `spool_path`. | Not covered by daemon gate; token should be mandatory for extension API. |
| Receiver archive state | `GET /v1/archive-state?provider=&provider_session_id=` | `server.py:202`, `receiver.py:357` | Same | Same | Spool/archive lifecycle for provider session, raw/index IDs, failure detail. | Not covered by daemon gate. |
| Receiver post-command poll | `GET /v1/post-commands` | `server.py:219`, `receiver.py:552` | Same | Same | Returns and by default claims pending extension commands. | Not covered; auth is required or a local process can drain commands. |
| Receiver capture ingest | `POST /v1/browser-captures` | `server.py:270`, `receiver.py:318` | Bearer only if configured; default no token | Origin allowlist; missing Origin allowed. No Host gate. | Writes capture JSON into spool. Body capped at 128 MiB; no total spool quota. | Not covered by daemon gate; token + spool governor required here. |
| Receiver post-command enqueue | `POST /v1/post-commands` | `server.py:263`, `receiver.py:520` | Bearer only if configured; env flag must enable posting | Same | Writes outbound post command to spool. | Not covered; token and quota needed. |
| Receiver post-command ack | `POST /v1/post-commands/:command_id/ack` | `server.py:266`, `receiver.py:587` | Bearer only if configured | Same | Mutates queued command status. | Not covered. |

### Threat pass by route class

For a hostile web page, the daemon GET surface is the critical read hole. A DNS-rebinding page can make the browser believe `http://attacker-host:port` is same-origin after the attacker’s DNS resolves to `127.0.0.1`. Because the daemon does not reject the foreign `Host`, normal browser same-origin protections do not save `GET /api/sessions`, `GET /api/sessions/:id`, `GET /api/events`, raw/provenance endpoints, `/metrics`, `/healthz/*`, or the HTML shells. Origin is not a replacement for Host on GET, and many benign GETs carry no Origin.

For hostile web-page mutation, the current POST/DELETE check is useful but too narrow. `_check_cross_origin()` accepts absent Origin at `http.py:1307-1309`, and the OTLP routes bypass the normal POST Origin check entirely at `http.py:1331-1354`. A same-origin rebinding POST is also not blocked by CORS; Host admission is the primary boundary.

For SSE, `GET /api/events` is authenticated only through the same global `_check_auth()` path as the other GET APIs. Because EventSource cannot set Authorization, `?access_token=` is accepted at `http.py:1045-1054`; however it is accepted for every authenticated route, not just `/api/events`. That means `/api/sessions?access_token=...` works today. The daemon disables normal access logging, but client-disconnect debug logs include `self.path` at `http.py:1299`, `1326`, and `1379`, so a URL token can still land in logs on broken pipes.

For a hostile local process, loopback is not a security boundary. Any same-machine process can set arbitrary `Host`, `Origin`, and `Sec-Fetch-*` headers. With no daemon token configured, the process can read archive APIs and mutate user state. With receiver auth unset, any process can POST captures, poll commands, or ack commands. A 0600 token file helps against other UIDs, but it does not protect against same-UID Linux processes; same UID can generally read the user’s config/token material, inspect process environment, and write to the same archive paths. Same-UID compromise should be treated as out of scope; other-UID and hostile-browser attacks are in scope.

For hostile captured content, most message text rendering is escaped, especially in `web_shell_reader.py:_polyBodyHtml()`. The risky sinks are not raw message bodies; they are ID/metadata values embedded into `innerHTML`, JavaScript handler attributes, and attachment-library HTML. `web_shell.py:411-412` defines `esc` and `escAttr`, but `escAttr` only handles quotes and does not handle backslashes or all HTML metacharacters for JavaScript string contexts. Examples include `onclick="selectSession('...')"` in `web_shell.py:1096-1117` and similar action rails in `web_shell_reader.py:200-223`. Attachment metadata has a clearer stored-XSS risk: `web_shell_attachments.py:373-402` builds attachment rows with partial escaping; `mime_type`, state/meta values, and origin can be inserted into `innerHTML` without robust escaping. Captured attachment MIME/origin fields are attacker-influenced enough to deserve a separate renderer hardening fix.

### Gaps in the planned kwsb.1 fix

The central daemon gate must run before every dispatch branch, not just before route-table APIs. Today `/`, `/s/:id`, `/w/:mode`, `/p`, `/a`, `/healthz/live`, `/healthz/ready`, and `/metrics` are reached before `_check_auth()` at `http.py:1193-1249`. A gate inserted only around `_static_get_routes()` and `_parameterized_get_routes()` would miss them.

The browser-capture receiver is a separate server. The bead design text says “all three holes live in `polylogue/daemon/http.py`” and points to `_authenticated_post_routes` as the seam for capture tokening. That is wrong for the receiver: the capture boundary is `polylogue/browser_capture/server.py` plus `polylogue/browser_capture/receiver.py`. Fixing daemon admission alone leaves `/v1/browser-captures`, `/v1/status`, `/v1/archive-state`, and `/v1/post-commands` exposed to local processes when no receiver token is configured.

The OTLP routes are special-cased before normal auth and Origin handling. When `observability_enabled` is true and the daemon bind host is loopback, `POST /v1/traces`, `/v1/metrics`, and `/v1/logs` accept telemetry without bearer or Origin checks. They also write raw payloads into `ops.db`. The admission gate needs an explicit observability policy, not a blind “POST routes are already covered” assumption.

`?access_token=` is too broad. It should only work for `GET /api/events` in SSE mode, or at most for `/api/events` regardless of poll/SSE shape. Every other route should require the Authorization header when a token is configured. The web shell should also set `Referrer-Policy: no-referrer` or equivalent on HTML responses if URL tokens remain necessary.

`Sec-Fetch-Site` is worth adding, but only as a browser-signal accelerator. A request with `Sec-Fetch-Site: cross-site` or `same-site` from a hostile page should be refused before route work. A missing `Sec-Fetch-Site` must not be treated as trusted browser traffic because curl, local exporters, old browsers, and scripts omit it. Host validation remains the hard boundary.

The receiver token lifecycle is not implemented as the bead describes. Current config has optional `browser_capture_auth_token`; the extension stores a manually entered token in `chrome.storage.local` and sends it as `Authorization` from `browser-extension/src/background.js:174-180`. I found no auto-minting, no 0600 receiver token file, no rotation path, and no `hmac.compare_digest` token compare. Extension storage is acceptable only as a client-side bearer cache, not as the root-of-trust store.

The spool governor needs precise semantics. The receiver has a per-request 128 MiB cap at `server.py:47` and `server.py:248`, but `write_capture_envelope()` at `receiver.py:318-341` has no total quota. Deterministic replacement of the same provider/session is good, but an attacker can vary `provider_session_id` and create many 128 MiB files. `post-commands` also writes unbounded JSON files under `receiver.py:493-548`. The correct behavior under quota should be: reject new growth with a retryable/storage-specific error, preserve existing files, allow same-artifact replacement when net growth fits quota, and never silently rotate away legitimate captures unless a documented retention policy exists.

Metrics and health are not content dumps, but they are still information disclosure and work triggers. `/metrics` exposes archive counts, DB sizes, storage layout, ingest status, and exception text as a label on collection failure. `/healthz/ready` can expose check messages and exception strings. Keep them unauthenticated for local probes if needed, but Host-gate them and redact exception text where possible.

There are reflection and rendering gaps outside kwsb.1. The admission fix stops remote web reads/writes, but hostile content already inside the archive can still reach shell `innerHTML` paths. That should be tracked separately from Host/Origin/token/spool, because it is a stored-content boundary rather than request-admission.

### Negative-test matrix

These are shaped to drop into the current in-process style used by `tests/unit/daemon/test_daemon_http_security.py` and `tests/unit/daemon/test_daemon_http_contracts.py`: parametrized, fixture-based, no live daemon socket. Receiver tests can reuse the existing `_running_receiver()` fixture style, though the matrix can also be adapted to direct handler construction.

| Test name | Request | Setup | Expected refusal / keep-working result |
|---|---|---|---|
| `test_foreign_host_rejected_for_shell_bootstrap` | `GET /` with `Host: evil.test:8766` | Loopback client, loopback bind, no token | `403 host_not_allowed`; shell handler not called. |
| `test_foreign_host_rejected_for_archive_gets` | Parametrize `GET /api/sessions`, `/api/sessions/s1`, `/api/sessions/s1/messages`, `/api/raw_artifacts/s1` with foreign Host | Token absent and token configured variants | Refused before archive read. With token configured, valid bearer still does not override bad Host. |
| `test_foreign_host_rejected_for_health_and_metrics` | `GET /healthz/live`, `/healthz/ready`, `/metrics` with `Host: evil.test` | No token | `403 host_not_allowed`. |
| `test_loopback_host_allows_shell_without_token` | `GET /`, `GET /s/s1`, `GET /w/stack`, `GET /p`, `GET /a` with `Host: 127.0.0.1:8766` or `localhost:8766` | Loopback bind/client | HTML response path reached. |
| `test_loopback_host_allows_health_and_metrics_without_token` | `GET /healthz/live`, `/healthz/ready`, `/metrics` with loopback Host | No token | Existing probe/scrape behavior remains allowed. |
| `test_get_api_still_requires_bearer_when_configured` | `GET /api/sessions` with loopback Host, no bearer | Server `auth_token="secret"` | `401 unauthorized`. |
| `test_query_access_token_rejected_outside_events` | `GET /api/sessions?access_token=secret`, `GET /api/status?access_token=secret` | Server `auth_token="secret"` | `401 unauthorized`; query token ignored. |
| `test_query_access_token_allowed_for_sse_events_only` | `GET /api/events?access_token=secret&max_seconds=1` | Server `auth_token="secret"`; patch event streaming to avoid sleeping | Route admitted. |
| `test_authorization_header_allowed_for_events_poll` | `GET /api/events?poll=1` with `Authorization: Bearer secret` | Server `auth_token="secret"` | Existing JSON poll keep-working case. |
| `test_cross_site_fetch_header_rejected_on_get` | `GET /api/sessions` with loopback Host and `Sec-Fetch-Site: cross-site` | No token or valid token | `403 cross_origin_denied` or `host_origin_denied`; archive handler not called. |
| `test_same_origin_loopback_fetch_header_allowed` | `GET /api/sessions` with loopback Host, `Origin: http://127.0.0.1:8766`, `Sec-Fetch-Site: same-origin` | Valid bearer if configured | Admitted. |
| `test_mutating_post_foreign_origin_rejected_even_with_token` | Parametrize `POST /api/reset`, `/api/ingest`, `/api/maintenance/plan`, `/api/maintenance/run`, `/api/user/marks`, `/api/user/annotations`, `/api/user/saved-views`, `/api/user/recall-packs`, `/api/user/workspaces` | Valid bearer, loopback Host, `Origin: https://evil.example` | `403 cross_origin_denied`; handler not called. |
| `test_mutating_delete_foreign_origin_rejected_even_with_token` | Parametrize all user-state DELETE routes | Valid bearer, loopback Host, foreign Origin | `403 cross_origin_denied`; user-state handler not called. |
| `test_browser_absent_origin_with_cross_site_fetch_rejected` | POST/DELETE route with no Origin but `Sec-Fetch-Site: cross-site` | Valid bearer | Refused; closes “Origin skipped when absent” for browser-like requests. |
| `test_local_tool_no_origin_still_allowed_with_bearer` | Same POST/DELETE routes with no Origin and no `Sec-Fetch-*` | Valid bearer, loopback Host | Existing curl/hook keep-working behavior remains allowed. |
| `test_otlp_foreign_host_rejected_before_observability_dispatch` | `POST /v1/traces`, `/v1/metrics`, `/v1/logs` with `Host: evil.test` | `observability_enabled=True` | `403 host_not_allowed`; `_handle_otlp_post` not called. |
| `test_otlp_cross_site_browser_request_rejected` | `POST /v1/traces` with loopback Host, `Origin: https://evil.example` or `Sec-Fetch-Site: cross-site` | `observability_enabled=True` | Refused before body parse/write. |
| `test_otlp_loopback_exporter_keep_working` | `POST /v1/traces` with loopback Host, no Origin, no Fetch headers | `observability_enabled=True`; body under `otlp_max_body_bytes` | Existing exporter path admitted. |
| `test_options_daemon_remains_closed` | `OPTIONS /api/sessions` | Any Host variant | Loopback Host returns existing 405; foreign Host returns Host refusal before method handling. |
| `test_all_registered_static_get_handlers_exist` | Iterate `_static_get_routes()` and `getattr(DaemonAPIHandler, handler_name)` | No network | Fails today on `_handle_provider_usage`; should pass after implementation or route removal. |
| `test_route_contracts_match_dispatch_handlers` | Compare `implemented_daemon_route_patterns()` with handler availability and route contracts | No network | Catches future registered-but-unimplemented routes. |
| `test_receiver_requires_token_for_capture_post` | `POST /v1/browser-captures` without Authorization | Receiver started with minted token required | `401 unauthorized`; no JSON file written. |
| `test_receiver_rejects_wrong_token_capture_post` | Same with `Authorization: Bearer wrong` | Token required | `401 unauthorized`; no file written. |
| `test_receiver_accepts_extension_origin_with_token` | Same with `Origin: chrome-extension://polylogue-test`, valid bearer | Token required | `202 accepted`; artifact written. |
| `test_receiver_rejects_local_process_spoof_without_token` | `POST /v1/browser-captures` with spoofed extension Origin but no bearer | Token required | `401 unauthorized`; demonstrates Origin alone is not auth. |
| `test_receiver_rejects_web_origin_without_allowed_origin` | `GET /v1/status` or capture POST with `Origin: https://chatgpt.com` | Default allowed origins | Existing `403 origin_not_allowed` remains. |
| `test_receiver_preflight_evil_origin_rejected` | `OPTIONS /v1/browser-captures` with `Origin: https://evil.example` | Token required | `403 origin_not_allowed`. |
| `test_receiver_preflight_extension_origin_allowed_without_bearer` | `OPTIONS /v1/browser-captures` with extension Origin | Token required | `204`; CORS preflight keep-working case. |
| `test_receiver_status_does_not_expose_absolute_spool_path_to_untrusted_clients` | `GET /v1/status` | Token required; maybe extension allowed | Without token refused; with token consider returning only readiness or relative/safe path. |
| `test_receiver_archive_state_requires_token` | `GET /v1/archive-state?provider=chatgpt&provider_session_id=conv` | Token required | Without token `401`; with token existing state DTO. |
| `test_receiver_post_command_poll_requires_token_and_does_not_claim_without_auth` | `GET /v1/post-commands` | Queued pending command, token required | Without token `401`; pending file remains pending. |
| `test_receiver_post_command_poll_keep_working_with_token` | Same with valid token | Posting enabled | Returns command and marks it dispatched as today. |
| `test_receiver_spool_quota_rejects_new_growth` | `POST /v1/browser-captures` with payload larger than remaining quota | Quota fixture: small max bytes, empty spool or nearly full spool | `507 insufficient_storage` or chosen typed quota status; no temp/orphan file remains. |
| `test_receiver_spool_quota_allows_same_artifact_replacement_when_net_growth_fits` | POST same provider/session twice | Quota near current file size | Replacement allowed if final total within quota. |
| `test_receiver_spool_quota_counts_post_commands` | `POST /v1/post-commands` with large text | Posting enabled, quota small | Refused with quota error; no command file created. |
| `test_receiver_token_compare_uses_constant_time_helper` | Unit test token checker | Patch/wrap helper or inspect behavior through exposed function | `_check_token` routes through `hmac.compare_digest` equivalent. |
| `test_receiver_minted_token_file_mode_0600` | Start/configure receiver without explicit token | Temp archive/config root | Token file created mode `0o600`; token loaded and required. |
| `test_receiver_token_rotation_invalidates_old_token` | Rotate token in fixture | Old and new bearer requests | Old token rejected, new token accepted. |
| `test_shell_escapes_session_id_in_js_handlers` | Render session list with ID containing backslash-quote payload | Direct render fixture | Output does not contain executable `onclick` break-out; preferably uses `data-*` plus event listeners or JSON encoding. |
| `test_attachment_library_escapes_mime_origin_and_meta` | Attachment item with MIME/origin containing `<img onerror=...>` | Direct JS/HTML render fixture | Rendered HTML contains escaped text, not raw tag/event handler. |
| `test_sse_url_token_not_logged_on_disconnect` | Simulate `GET /api/events?access_token=secret` broken pipe | Patch logger | Log record path has token redacted or absent. |

### Fix ordering

First, add a single daemon request-admission function and call it at the top of `do_GET`, `_do_post_impl`, `_do_delete_impl`, and `do_OPTIONS`, before any shell, health, metrics, OTLP, static, parameterized, or user-state dispatch. It should validate Host for all routes, use Origin and `Sec-Fetch-Site` for browser-origin signals, and preserve no-Origin local-tool behavior only when there are no browser fetch headers.

Second, patch the separate receiver server. Make bearer auth default-on via an auto-minted token, compare with `hmac.compare_digest`, require token on GET and POST data routes, keep preflight origin-only, and add the spool governor at the writer functions, not just HTTP body parsing.

Third, restrict `?access_token=` to `/api/events`, redact URL tokens from logs, and add a route-contract test proving every registered handler exists. That immediately catches the current `GET /api/provider-usage` registration bug.

Fourth, file the renderer hardening separately: replace JS string interpolation in `onclick` attributes with event listeners and data attributes, and use one context-correct escaping path for attachment metadata.
