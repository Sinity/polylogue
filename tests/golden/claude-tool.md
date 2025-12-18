---
attachmentPolicy:
  charThreshold: 4000
  extractedCount: 2
  lineThreshold: 40
  previewLines: 5
  routing:
    routed: 0
    skipped: 2
attachment_bytes: 295
attachmentsDir: <path>
collapseThreshold: 10
contentHash: 419d733be8531e4ca97dfc8a943a0f7c70af11804f4d0cafecb5c311d502ec6f
currentBranch: branch-000
dirty: false
html: false
htmlPath: null
lastImported: '2000-01-01T00:00:00Z'
lastUpdated: '2024-01-01T00:00:03Z'
outputPath: <path>
polylogue:
  contentHash: 419d733be8531e4ca97dfc8a943a0f7c70af11804f4d0cafecb5c311d502ec6f
  conversationId: golden-claude-tool-1
  lastUpdated: '2024-01-01T00:00:03Z'
  provider: claude.ai
  slug: golden-claude-tool-attachments
  title: Golden Claude Tool + Attachments
slug: golden-claude-tool-attachments
sourceExportPath: tests/fixtures/golden/claude_tool
sourceModel: claude-3.5-sonnet
title: Golden Claude Tool + Attachments
token_count: 28
tokens: 28
word_count: 28
words: 28
---

## User

Hello. Please run a command and show the result. Also see attachments.

## Model

Tool call `bash`
```json
{
  "cmd": "echo hi"
}
```

Tool result
````
hi
````

Done.