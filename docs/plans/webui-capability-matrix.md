# Typed WebUI capability matrix

This inventory is the cutover ledger for independently valuable browser
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
| Costs and unknown values | `/cost` | shell cost panel | typed |
| Realtime and observability | `/observability` | shell status panels | typed |
| Generated contracts and asset manifest | `webui/src/api`, daemon bundle | inline script assembly | typed |
| Authentication and CSP boundary | daemon web-auth/asset routes | shell bootstrap | typed |
| Topology and provenance | no typed route | shell panels/API | legacy |
| Attachments and pastes | no typed route | `/a`, `/p`, shell reader | legacy |
| Assertions and selection | no typed route | shell toolbar/API | legacy |
| Compare, similar, and workspace | no typed route | `/w`, shell panels | legacy |
| Keyboard, responsive, and design-system states | component/browser suites | shell interaction tests | typed |

There are no unclassified capability cells in this inventory. Legacy cells are
explicit cutover work; they are not evidence that the typed surface has parity.
The root and `/app` are aliases into the same typed SSR handlers, so they do not
constitute separate browser shells.
