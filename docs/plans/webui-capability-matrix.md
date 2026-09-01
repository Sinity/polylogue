# Typed WebUI capability matrix

This is the current cutover ledger for independently valuable browser
capabilities. `typed` means the canonical typed surface serves the capability;
`legacy` means the capability still depends on the interpolated reader;
`superseded` means the typed interaction replaces the old implementation.

| Capability | Typed surface | Legacy surface | Cutover state |
| --- | --- | --- | --- |
| Archive overview | `/` | `/` | typed |
| Bounded search | `/search` | shell search | typed |
| Session list and filters | `/sessions` | shell list | typed |
| Bounded session transcript | `/sessions/:id`, `/s/:id` | shell reader | typed |
| Semantic message cards | session read island | shell semantic cards | typed |
| Topology | session read API and typed session route | shell lineage panels/API | typed envelope |
| Provenance and raw evidence | session read API and typed session route | shell provenance/raw panels/API | typed envelope |
| Attachments | `/a`, typed semantic metadata projection, API | shell attachment panel/API | typed |
| Pastes | `/p`, typed semantic evidence projection, API | shell paste panel/API | typed |
| Costs and unknown values | `/cost` | shell cost panel | typed |
| Assertions | no typed route | shell notes and assertion API | legacy |
| Session selection | no typed route | shell selection toolbar/API | legacy |
| Saved views | no typed route | shell saved-view controls/API | legacy |
| Workspace and stack views | `/w/:mode`, typed semantic projection, API | shell workspace panel | typed |
| Compare | `/w/compare`, typed semantic projection, API | shell compare panel | typed |
| Similar-session exploration | no typed route | shell similar panel/API | legacy |
| Observability | `/observability` | shell status panels | typed |
| Realtime convergence updates | no typed route | shell realtime client | legacy |
| Generated contracts and asset manifest | `webui/src/api`, daemon bundle | inline script assembly | typed |
| Authentication and CSP boundary | daemon web-auth/asset routes | shell bootstrap | typed |
| Keyboard, responsive, and design-system states | component/browser suites | shell interaction tests | typed |

The root, `/app`, `/w/*`, `/p`, and `/a` routes enter typed SSR handlers and
share the daemon's asset-manifest and authentication boundary. The legacy
shell modules remain internal compatibility code until the remaining reader
tests and route helpers are retired in the final deletion slice.
