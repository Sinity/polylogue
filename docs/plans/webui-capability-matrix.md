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
| Topology | no typed route | shell lineage panels/API | legacy |
| Provenance and raw evidence | no typed route | shell provenance/raw panels/API | legacy |
| Attachments | no typed route | `/a`, shell attachment panel/API | legacy |
| Pastes | no typed route | `/p`, shell paste panel/API | legacy |
| Costs and unknown values | `/cost` | shell cost panel | typed |
| Assertions | no typed route | shell notes and assertion API | legacy |
| Session selection | no typed route | shell selection toolbar/API | legacy |
| Saved views | no typed route | shell saved-view controls/API | legacy |
| Workspace and stack views | no typed route | `/w`, shell workspace panel | legacy |
| Compare | no typed route | `/w/compare`, shell compare panel | legacy |
| Similar-session exploration | no typed route | shell similar panel/API | legacy |
| Observability | `/observability` | shell status panels | typed |
| Realtime convergence updates | no typed route | shell realtime client | legacy |
| Generated contracts and asset manifest | `webui/src/api`, daemon bundle | inline script assembly | typed |
| Authentication and CSP boundary | daemon web-auth/asset routes | shell bootstrap | typed |
| Keyboard, responsive, and design-system states | component/browser suites | shell interaction tests | typed |

Legacy cells are explicit cutover work. The root and `/app` are aliases into
the same typed SSR handlers, so they do not constitute separate browser shells.
