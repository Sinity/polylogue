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
| Assertions | `/api/assertions` typed envelope | notes and assertion API | typed envelope |
| Session selection | `/api/user/marks` typed operation | selection toolbar/API | typed operation |
| Saved views | `/api/user/saved-views` typed operation | saved-view controls/API | typed operation |
| Workspace and stack views | `/w/:mode`, typed semantic projection, API | shell workspace panel | typed |
| Compare | `/w/compare`, typed semantic projection, API | shell compare panel | typed |
| Similar-session exploration | `/api/similar` typed envelope | similar panel/API | typed envelope |
| Observability | `/observability` | shell status panels | typed |
| Realtime convergence updates | no typed route | shell realtime client | legacy |
| Generated contracts and asset manifest | `webui/src/api`, daemon bundle | inline script assembly | typed |
| Authentication and CSP boundary | daemon web-auth/asset routes | shell bootstrap | typed |
| Keyboard, responsive, and design-system states | component/browser suites | shell interaction tests | typed |

All browser routes enter typed SSR handlers and share the daemon's
asset-manifest and authentication boundary. Mutations remain typed API
operations and are not embedded in browser rendering.
