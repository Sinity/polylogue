# Polylogue Search & Discovery Roadmap

## Overview

This roadmap outlines 15 creative search and discovery features to dramatically improve how users navigate and rediscover their conversations. Current search is limited to basic FTS5 keyword matching and optional Qdrant semantic search. These enhancements enable powerful discovery patterns: temporal queries, attachment search, similar conversations, topic clustering, and interactive browsing.

**Current State**:
- FTS5 keyword search with basic query escaping
- Optional Qdrant semantic search (Voyage embeddings)
- Search result caching (21,343x speedup)
- Limited temporal support (`--since` ISO format only)
- No result previews or context

---

## Tier 1: High-Impact, Lower Effort

### 1. Hybrid Search with Reciprocal Rank Fusion

**Files Affected**: `storage/search_providers/fts5.py`, `storage/search_providers/qdrant.py`, `protocols.py`

**Problem**: FTS5 and Qdrant operate independently. Keyword matches may miss semantically similar content; semantic search may ignore exact matches.

**Solution**: Implement Reciprocal Rank Fusion (RRF) to combine both ranking systems:

```python
def reciprocal_rank_fusion(fts_results: list[SearchResult],
                          semantic_results: list[SearchResult],
                          k: int = 60) -> list[SearchResult]:
    """Combine rankings using RRF formula: 1/(k + rank)"""
    fts_scores = {r.message_id: 1.0 / (k + i) for i, r in enumerate(fts_results)}
    semantic_scores = {r.message_id: 1.0 / (k + i) for i, r in enumerate(semantic_results)}

    # Merge with harmonic mean of scores
    combined = {}
    all_ids = set(fts_scores.keys()) | set(semantic_scores.keys())

    for msg_id in all_ids:
        fts_score = fts_scores.get(msg_id, 0)
        sem_score = semantic_scores.get(msg_id, 0)
        # Harmonic mean weights both equally
        combined[msg_id] = 2 * fts_score * sem_score / (fts_score + sem_score) if (fts_score + sem_score) > 0 else 0

    # Re-rank and return
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [result_by_id[mid] for mid, _ in ranked[:len(fts_results)]]

class HybridSearchProvider(SearchProvider):
    def __init__(self, fts5: FTS5Provider, qdrant: QdrantProvider):
        self.fts5 = fts5
        self.qdrant = qdrant

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search both providers, fuse results."""
        fts_results = self.fts5.search(query, limit=limit * 2)
        semantic_results = self.qdrant.search(query, limit=limit * 2)
        return reciprocal_rank_fusion(fts_results, semantic_results, k=60)[:limit]
```

**Implementation Locations**:
- `storage/search_providers/hybrid.py` (new)
- Update `container.py` to instantiate HybridSearchProvider when both enabled

**User Value**: "Find conversations about Python error handling" returns both exact keyword matches AND semantically similar discussions about exception patterns. Dramatically improves recall.

**Success Criteria**:
- Hybrid results include top keyword and top semantic results
- User satisfaction: +30% on relevance surveys

---

### 2. Natural Language Temporal Search

**Files Affected**: `storage/search.py`, `cli/commands/search.py`, new `util/temporal.py`

**Current Limitation**: `--since` only accepts ISO format (`2024-01-15`). Users must remember exact dates.

**Solution**: Add natural language date parsing via `dateparser` library:

```python
from dateparser import parse as parse_date

def parse_temporal_expr(expr: str) -> tuple[datetime | None, datetime | None]:
    """Parse natural language temporal expressions."""
    expressions = {
        "last week": (datetime.now() - timedelta(days=7), datetime.now()),
        "last month": (datetime.now() - timedelta(days=30), datetime.now()),
        "last year": (datetime.now() - timedelta(days=365), datetime.now()),
        "2024-Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
        "2024-Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
        "january": (datetime(datetime.now().year, 1, 1), datetime(datetime.now().year, 1, 31)),
    }

    if expr in expressions:
        return expressions[expr]

    # Fallback to dateparser for other formats
    parsed = parse_date(expr)
    if parsed:
        return parsed, None  # Single date, no end

    raise ValueError(f"Could not parse temporal expression: {expr}")
```

**CLI Enhancement**:
```
polylogue search "python error" --since "last month"
polylogue search "rust" --during "2024-Q1"
polylogue search "debugging" --around "2024-06-15" --window 7d
polylogue search --created "last week"  # Only conversations created in last week
```

**Implementation**:
- Add `dateparser>=1.1.0` to `pyproject.toml`
- Update `cli/commands/search.py` to parse temporal expressions
- Add validation in `SearchFilters` dataclass

**Success Criteria**:
- Parse 20+ common temporal expressions
- Zero ambiguity in quarter/month parsing

---

### 3. Search Result Previews with Context Windows

**Files Affected**: `storage/search.py` (SearchResult), `cli/commands/search.py`, `ui/facade.py`

**Problem**: Current results show only the matching line. User must open full conversation to understand relevance.

**Solution**: Enhance SearchResult with surrounding context:

```python
@dataclass
class SearchResult:
    message_id: str
    conversation_id: str
    score: float
    text_preview: str  # Matching excerpt
    context_before: str | None = None  # Previous message
    context_after: str | None = None   # Following message
    match_count: int = 1  # How many times in this conversation
    adjacent_role: Literal["user", "assistant", "both"] | None = None
```

**Implementation**:
```python
def search_with_context(self, query: str, limit: int = 20) -> list[SearchResult]:
    """Search with KWIC (Key Word In Context) rendering."""
    base_results = self.fts5.search(query, limit)

    enhanced = []
    for result in base_results:
        # Load conversation
        conv = self.repository.get_conversation(result.conversation_id)

        # Find message
        message = next((m for m in conv.messages if m.message_id == result.message_id), None)
        if not message:
            continue

        # Get surrounding messages
        idx = conv.messages.index(message)
        context_before = conv.messages[idx-1].text[:200] if idx > 0 else None
        context_after = conv.messages[idx+1].text[:200] if idx < len(conv.messages)-1 else None

        # Count matches in conversation
        match_count = sum(1 for m in conv.messages if query.lower() in m.text.lower())

        enhanced.append(SearchResult(
            message_id=result.message_id,
            conversation_id=result.conversation_id,
            score=result.score,
            text_preview=_highlight_match(message.text, query),
            context_before=context_before,
            context_after=context_after,
            match_count=match_count,
            adjacent_role=conv.messages[idx-1].role if idx > 0 else None
        ))

    return enhanced
```

**CLI Display**:
```
Query: "python error handling"

> Conv abc123: 2 matches
  Before:  "Let me think about this..."
  Match:   "I'm getting a >>> KeyError in my error handling code
  After:   "Ah, I see the issue now"

  Score: 0.85 | 2 matches in conversation
```

**Success Criteria**:
- Show 2-3 lines of context (before/after)
- Highlight matching phrase in preview
- Users can understand relevance without opening full conversation

---

### 4. Attachment-Aware Search

**Files Affected**: `storage/search.py`, `storage/store.py`, `cli/commands/search.py`

**Problem**: No way to find "conversations where I shared a PDF" or "discussions with screenshots"

**Solution**: Index attachment metadata alongside messages:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS attachments_fts USING fts5(
    attachment_id UNINDEXED,
    conversation_id UNINDEXED,
    filename,
    path,
    mime_type UNINDEXED
);
```

**CLI Enhancements**:
```bash
polylogue search --has-attachments  # Any attachments
polylogue search --mime-type "image/*"  # Images only
polylogue search --mime-type "application/pdf"  # PDFs
polylogue search --filename "*.csv"  # CSV files
polylogue search "data analysis" --has-attachments  # Keyword + attachments
```

**Implementation**:
```python
class SearchFilters:
    has_attachments: bool = False
    mime_type_pattern: str | None = None  # Supports wildcards

def search_with_filters(self, query: str, filters: SearchFilters) -> list[SearchResult]:
    """Search with attachment filtering."""
    results = self.fts5.search(query)

    if filters.has_attachments or filters.mime_type_pattern:
        # Filter to conversations with matching attachments
        matching_convs = set()
        for conv_id, attachments in self._iter_conversation_attachments():
            if filters.has_attachments and attachments:
                matching_convs.add(conv_id)
            elif filters.mime_type_pattern:
                for att in attachments:
                    if fnmatch.fnmatch(att.mime_type, filters.mime_type_pattern):
                        matching_convs.add(conv_id)

        results = [r for r in results if r.conversation_id in matching_convs]

    return results
```

**Success Criteria**:
- Find conversations with attachments: <100ms
- Filter by mime type: Accurate matching
- Support 10+ common patterns (pdf, image/*, *.csv)

---

### 5. Saved Searches & Search History

**Files Affected**: New `storage/saved_searches.py`, `cli/commands/search.py`, `config.py`

**Problem**: Power users repeat the same queries (e.g., "python error handling", "rust async"). No way to save and reuse.

**Solution**: Store searches with optional tags:

```python
@dataclass
class SavedSearch:
    name: str  # Unique identifier
    query: str  # Search query
    filters: dict[str, Any]  # Serialized SearchFilters
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime | None = None
    use_count: int = 0

# Database table
CREATE TABLE IF NOT EXISTS saved_searches (
    name TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    filters JSON NOT NULL,
    tags TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    use_count INTEGER DEFAULT 0
);

# Search history
CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    result_count INTEGER
);
```

**CLI Integration**:
```bash
# Save a search
polylogue search "python error handling" --save "python-errors"

# List saved searches
polylogue search --list-saved
# Output:
# - python-errors (5 uses, last: 2024-06-15)
# - rust-async (2 uses, last: 2024-06-10)

# Run saved search
polylogue search --run "python-errors"

# View search history
polylogue search --history | head -20

# Fuzzy search history (with fzf)
polylogue search --history | fzf | xargs polylogue search
```

**Implementation**:
- New `SavedSearchRepository` for CRUD operations
- CLI command: `search --save <name>`, `--run <name>`, `--list-saved`, `--history`

**Success Criteria**:
- Save 50+ searches without performance degradation
- Fuzzy search history with fzf integration
- Persist across sessions

---

## Tier 2: Medium-Impact Discovery Features

### 6. Conversation Similarity Graph

**Files Affected**: New `discovery/similarity.py`, `cli/commands/similar.py`, `storage/search_providers/qdrant.py`

**Problem**: "I had a great conversation about X, show me related ones I might have forgotten"

**Solution**: Use Qdrant embeddings to find similar conversations:

```python
class SimilarityFinder:
    def __init__(self, qdrant_provider: QdrantProvider, repository: StorageRepository):
        self.qdrant = qdrant_provider
        self.repository = repository

    def find_similar(self, conversation_id: str, limit: int = 10) -> list[tuple[ConversationId, float, str]]:
        """Find conversations semantically similar to given conversation."""
        conv = self.repository.get_conversation(conversation_id)

        # Compute conversation-level embedding (average message embeddings)
        message_embeddings = self.qdrant.get_embeddings([m.message_id for m in conv.messages])
        conv_embedding = np.mean(message_embeddings, axis=0)

        # Search for similar conversations
        similar = self.qdrant.search(
            query_vector=conv_embedding,
            limit=limit + 1,  # +1 to exclude self
            query_filter=models.Filter(must_not=[
                models.HasIdCondition(has_id=[conversation_id])
            ])
        )

        # Return similar conversations with titles
        results = []
        for result in similar[:limit]:
            similar_conv = self.repository.get_conversation(result.message_id.split(":")[0])
            results.append((
                similar_conv.conversation_id,
                result.score,
                similar_conv.title
            ))

        return results
```

**CLI Command**:
```bash
polylogue similar abc123 --limit 5
# Output:
# 1. "Debugging async Python issues" (0.89 similarity)
# 2. "Python threading patterns" (0.87 similarity)
# 3. "Async/await deep dive" (0.84 similarity)

# Open in browser
polylogue similar abc123 --open
```

**Success Criteria**:
- Find 5 similar conversations: <500ms
- Similarity threshold: >0.75 (meaningful similarity)

---

### 7. Topic Clustering & Auto-Tagging

**Files Affected**: New `discovery/clustering.py`, `lib/models.py` (add topics field), storage migration

**Problem**: Browse conversations by emergent topics rather than just providers

**Solution**: Run clustering on Qdrant embeddings:

```python
class TopicExtractor:
    def __init__(self, qdrant: QdrantProvider, repository: StorageRepository):
        self.qdrant = qdrant
        self.repository = repository

    def cluster_conversations(self, min_cluster_size: int = 5) -> dict[str, list[str]]:
        """Cluster conversations by semantic similarity, extract labels."""
        # Get all conversation embeddings
        all_convs = list(self.repository.iter_conversations())
        embeddings = np.array([
            self._get_conv_embedding(conv) for conv in all_convs
        ])

        # Clustering with HDBSCAN
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)

        # Group conversations by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise point
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_convs[i].conversation_id)

        # Extract topic labels via TF-IDF
        topics = {}
        for label, conv_ids in clusters.items():
            messages = []
            for cid in conv_ids:
                conv = self.repository.get_conversation(cid)
                messages.extend([m.text for m in conv.messages])

            # Extract top TF-IDF terms
            vectorizer = TfidfVectorizer(max_features=3, stop_words='english')
            tfidf = vectorizer.fit_transform(messages)
            terms = vectorizer.get_feature_names_out()
            topics[label] = ", ".join(terms)

        return topics

    def add_auto_tags(self) -> None:
        """Add computed topics to all conversations."""
        topics = self.cluster_conversations()

        for conv_id, topic_label in topics.items():
            self.repository.update_conversation(conv_id, {
                'auto_topics': topic_label.split(", "),
                'primary_topic': topic_label.split(",")[0]
            })
```

**CLI Integration**:
```bash
# Cluster and extract topics (one-time)
polylogue analyze --cluster

# Browse by topic
polylogue browse --topic "python"
polylogue browse --topic "rust" --limit 20

# List all topics
polylogue topics --count
# Output:
# python (45 conversations)
# rust (23 conversations)
# web (18 conversations)
```

**Success Criteria**:
- Extract 15-30 topics from typical archive
- Topics are meaningful and interpretable

---

### 8. Timeline/Calendar View

**Files Affected**: New `cli/commands/timeline.py`, `ui/calendar.py`, `rendering/renderers/calendar.py`

**Problem**: Understand conversation patterns over time, find "that conversation from early June"

**Solution**: Aggregate conversations by time periods:

```bash
polylogue timeline --month 2024-06
# Output:
# June 2024 Activity
# ==================
#
#   1  2  3  4  5  6  7
#   .  .  3  1  .  4  2
#   1  3  .  2  1  5  3
#   2  1  4  .  3  2  1
#   4  .  2  3  .  1  4
#
# Top topics: ["python", "nix", "shell scripting"]
# Total: 42 conversations, 1,234 messages
#
# Heatmap by hour:
#   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#   .  .  .  .  .  .  1  2  5  8 12 15 18 14 12  9  6  4  2  1  .  .  .  .
```

**Implementation**:
```python
class TimelineView:
    def render_monthly(self, year: int, month: int) -> str:
        """Render calendar heatmap."""
        conversations = list(self.repository.iter_conversations(
            after=datetime(year, month, 1),
            before=datetime(year, month + 1, 1)
        ))

        # Group by day
        by_day = defaultdict(list)
        for conv in conversations:
            day = conv.created_at.day
            by_day[day].append(conv)

        # Render ASCII calendar
        cal = calendar.monthcalendar(year, month)
        output = []
        for week in cal:
            for day in week:
                if day == 0:
                    output.append(".")
                else:
                    count = len(by_day[day])
                    output.append(str(count) if count > 0 else ".")

        return output

    def render_hour_heatmap(self, date: datetime) -> str:
        """Heatmap by hour of day."""
        messages = self.repository.iter_messages(
            after=datetime(date.year, date.month, date.day),
            before=datetime(date.year, date.month, date.day + 1)
        )

        # Group by hour
        by_hour = defaultdict(int)
        for msg in messages:
            hour = msg.timestamp.hour
            by_hour[hour] += 1

        # Render
        for hour in range(24):
            count = by_hour[hour]
            bar_width = count // 5  # Scale
            print(f"{hour:2d}: {'█' * bar_width}")
```

**Success Criteria**:
- Render 12-month view: <1 second
- Interactive navigation (prev/next month)
- Export to iCal format

---

### 9. Thinking Trace Search

**Files Affected**: `lib/models.py` (is_thinking), new `cli/commands/thinking.py`, storage index

**Problem**: "How did Claude reason about this problem?" - need to search specifically in thinking blocks

**Solution**: Enable reasoning-specific search:

```bash
# Search only in thinking blocks
polylogue search "factorial" --thinking-only

# View thinking traces
polylogue view --projection thinking --since "last week"

# Compare thinking patterns
polylogue analyze --compare-thinking "python" "rust"
```

**Implementation**:
```python
# Add to FTS5 index for thinking content
CREATE VIRTUAL TABLE IF NOT EXISTS thinking_fts USING fts5(
    message_id UNINDEXED,
    conversation_id UNINDEXED,
    thinking_text
);

# Query
def search_thinking(self, query: str, limit: int = 20) -> list[SearchResult]:
    """Search only in thinking blocks."""
    conn = self.connection()
    results = conn.execute("""
        SELECT message_id, conversation_id, thinking_text
        FROM thinking_fts
        WHERE thinking_fts MATCH ?
        LIMIT ?
    """, (escape_fts5_query(query), limit)).fetchall()
    return [SearchResult(...) for row in results]
```

**Success Criteria**:
- Find reasoning patterns across conversations
- Thinking-only search returns only thinking blocks

---

### 10. Faceted Search Navigation

**Files Affected**: `storage/search.py`, `cli/commands/search.py`

**Problem**: See distribution of results before drilling down

**Solution**: Return facet counts with search results:

```python
@dataclass
class FacetedSearchResult:
    hits: list[SearchResult]
    facets: dict[str, dict[str, int]] = field(default_factory=dict)
    # Example:
    # {
    #   "provider": {"claude": 10, "chatgpt": 5, "gemini": 2},
    #   "has_thinking": {"yes": 12, "no": 5},
    #   "has_code": {"yes": 8, "no": 9},
    #   "year": {"2024": 15, "2023": 2}
    # }
```

**CLI**:
```bash
polylogue search "error" --facets
# Results: 17 conversations
#
# Providers:     Claude: 10 | ChatGPT: 5 | Gemini: 2
# Has Thinking:  Yes: 12 | No: 5
# Has Code:      Yes: 8 | No: 9
# Year:          2024: 15 | 2023: 2
```

**Success Criteria**:
- Compute facets: <200ms
- Support 5+ facet dimensions

---

## Integration Points

### Search Improvements Enabled
- Hybrid search unlocks semantic + keyword recall
- Temporal operators enable time-based navigation
- Context windows provide quick relevance assessment
- Attachment search enables file-based discovery

### UI Layers
- CLI: Enhanced `search` and `view` commands
- TUI: Interactive result browser with live preview
- Web: Search sidebar with facet filters
- API: `/api/search?q=...&facets=true`

---

## Priority Matrix

| # | Feature | Impact | Effort | ROI | Users |
|---|---------|--------|--------|-----|-------|
| 1 | Hybrid Search | High | Low | 10:1 | Frequent searchers |
| 2 | Natural Temporal | High | Low | 8:1 | Chronological users |
| 3 | Context Preview | High | Low | 7:1 | Result validators |
| 4 | Attachment Search | Medium | Low | 6:1 | File sharers |
| 5 | Saved Searches | Medium | Low | 5:1 | Power users |
| 6 | Similarity Graph | Medium | Medium | 4:1 | Explorers |
| 7 | Topic Clustering | Medium | Medium | 4:1 | Analysts |
| 8 | Timeline View | Medium | Medium | 3:1 | Chroniclers |
| 9 | Thinking Search | Medium | Low | 4:1 | Researchers |
| 10 | Faceted Search | Medium | Low | 5:1 | Browsers |

---

## Success Metrics

- **Discovery**: Users find relevant conversations 3x faster
- **Precision**: Top-5 search results relevant 90% of the time
- **Recall**: Saved searches enable reuse of 60% of common queries
- **Engagement**: 40% of searches use facets or filters
- **Satisfaction**: "I can find what I'm looking for" score: 8/10

---

## Implementation Order

**Phase 1 (Weeks 1-2)**: High-ROI quick wins
- Hybrid search
- Natural temporal operators
- Context previews
- Attachment search

**Phase 2 (Weeks 3-4)**: User convenience
- Saved searches
- Search history
- Faceted navigation

**Phase 3 (Weeks 5-6)**: Discovery features
- Similarity graph
- Topic clustering
- Timeline view
- Thinking trace search
