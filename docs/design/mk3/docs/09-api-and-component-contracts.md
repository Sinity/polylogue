# MK3 API and component contracts

## Existing HTTP base to preserve

Preserve the current daemon read surface shape:

- `/api/health`
- `/api/status`
- `/api/conversations`
- `/api/conversations/{id}`
- `/api/conversations/{id}/messages`
- `/api/conversations/{id}/raw`
- `/api/raw_artifacts/{id}`
- `/api/facets`
- `/api/sources`
- `/api/maintenance/plan`
- `/api/maintenance/run`

## New read endpoints

Conversation enrichment:

```text
GET /api/conversations/{id}/outline
GET /api/conversations/{id}/attachments
GET /api/conversations/{id}/pastes
GET /api/conversations/{id}/topology
GET /api/conversations/{id}/provenance
GET /api/conversations/{id}/insights
```

Archive-wide object browsers:

```text
GET /api/attachments?query=&mime=&state=&provider=&limit=&cursor=
GET /api/pastes?query=&hash=&provider=&min_lines=&limit=&cursor=
GET /api/topology?conversation_id=&edge_type=&confidence=&limit=&cursor=
GET /api/insights?kind=&conversation_id=&status=&limit=&cursor=
```

Stack/compare:

```text
GET /api/stack?ids=convA,convB,convC
GET /api/compare?left=convA&right=convB&align=prompt|time|fork_point|search
```

Realtime:

```text
GET /api/events?topics=conversation.*,message.appended,insights.refreshed
```

Server event envelope:

```json
{
  "event_id": "monotonic or uuid",
  "event_type": "conversation.created | conversation.updated | message.appended | insights.refreshed | fts.repaired | cost.updated | snapshot",
  "topic": "conversation:<id> or archive",
  "occurred_at": "iso",
  "payload": {},
  "coalesced": false,
  "snapshot_required": false
}
```

## User-state endpoints

```text
GET    /api/user/marks?target_type=&target_id=&mark_type=&conversation_id=&message_id=
POST   /api/user/marks
DELETE /api/user/marks?target_type=&target_id=&mark_type=&conversation_id=&message_id=

GET    /api/user/annotations?target_type=&target_id=
POST   /api/user/annotations
GET    /api/user/annotations/{annotation_id}
DELETE /api/user/annotations/{annotation_id}

GET    /api/user/saved-views
POST   /api/user/saved-views
GET    /api/user/saved-views/{view_id}
DELETE /api/user/saved-views/{view_id}

GET    /api/user/recall-packs
POST   /api/user/recall-packs
GET    /api/user/recall-packs/{pack_id}
DELETE /api/user/recall-packs/{pack_id}

GET    /api/user/workspaces
POST   /api/user/workspaces
GET    /api/user/workspaces/{workspace_id}
DELETE /api/user/workspaces/{workspace_id}
```

Mutation envelope:

```json
{
  "status": "ok | no_op | not_found | conflict | error",
  "outcome": "created | updated | deleted | already_present | not_present | unchanged",
  "target": {"target_type":"message", "target_id":"..."},
  "affected_count": 1,
  "detail": "stable machine detail",
  "error": null
}
```

## Component list

Shell:

- `StatusStrip`
- `GlobalSearchBox`
- `CommandPalette`
- `WorkspaceTabs`
- `PanelResizeHandle`
- `KeyboardHelp`

Search:

- `FilterChips`
- `FacetPanel`
- `SavedViewList`
- `SearchResultRow`
- `HitExplanation`
- `BulkSelectionBar`

Conversation:

- `ConversationHeader`
- `ConversationStatusChips`
- `MessageVirtualList`
- `MessageCard`
- `MessageActionRail`
- `ContentSegment`
- `PasteBlock`
- `AttachmentCard`
- `ToolBlock`
- `ThinkingBlock`
- `CopyMenu`

Inspector:

- `InspectorShell`
- `OutlinePanel`
- `RawPanel`
- `ProvenancePanel`
- `NotesPanel`
- `TopologyPanel`
- `AttachmentsPanel`
- `InsightsPanel`

Workspace:

- `TabStrip`
- `StackLane`
- `CompareLane`
- `TimelineEvent`
- `WorkspaceSaveDialog`

Object browsers:

- `AttachmentLibrary`
- `PasteLibrary`
- `TopologyExplorer`
- `InsightsBrowser`

State components:

- `DegradedBanner`
- `EmptyState`
- `PartialDataChip`
- `LiveStatusChip`
- `MutationToast`
- `DisabledReasonTooltip`
