# Polylogue Data Model & Content Enrichment Roadmap

## Overview

This roadmap outlines 15 data model enhancements to enable richer semantic understanding of conversations. Current content_blocks extraction captures thinking/tool_use/tool_result; this roadmap extends the model to handle code, citations, math, structured data, and computed metadata (topics, complexity, domain classification).

**Current State**:
- Message-level content_blocks: text, thinking, tool_use, tool_result
- Conversation-level metadata: provider, title, timestamps, message_count
- Message properties: is_thinking, is_tool_use, is_substantive
- No cross-conversation linking, no computed metadata

---

## Priority 1: Extended Content Block Types

### 1. Code Block with Language Detection

**Files Affected**: `lib/models.py` (Message.content_blocks), importers (all), `storage/store.py`

**Current State**: Code appears as plain text within `type: "text"` blocks, losing semantic information.

**Enhancement**:
```python
@dataclass
class CodeContentBlock(ContentBlock):
    type: Literal["code"] = "code"
    language: str  # "python", "rust", "sql", "bash"
    text: str      # Code content
    filename: str | None = None  # Optional: "main.py"
    line_range: tuple[int, int] | None = None  # Extracted code line numbers
    syntax_valid: bool | None = None  # Basic validation
```

**Extraction Rules** (per provider):
- **ChatGPT**: Detect fenced blocks (```python), code_interpreter outputs
- **Claude**: Structured blocks already in content array
- **Claude Code**: tool_use with Bash/Read/Write/EditStr → code blocks
- **Gemini**: Extract from structured code samples

**Implementation**:
```python
# In each importer
def extract_code_blocks(content: str | list) -> list[ContentBlock]:
    blocks = []
    if isinstance(content, str):
        # Regex for ```lang ... ``` patterns
        for match in re.finditer(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL):
            lang = match.group(1) or "text"
            code = match.group(2)
            blocks.append(CodeContentBlock(
                language=lang,
                text=code,
                syntax_valid=validate_syntax(code, lang)
            ))
    return blocks

# Storage
CREATE TABLE code_blocks (
    code_block_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    filename TEXT,
    syntax_valid BOOLEAN,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);
```

**User Value**:
- Syntax-highlighted rendering in HTML/TUI output
- Language-specific search ("show me Python I discussed")
- Analytics: Percentage of conversations with code by language

---

### 2. Citation/Reference Block

**Files Affected**: `lib/models.py`, importers, new `discovery/citations.py`

**Problem**: No structured way to track when AI cites sources (docs, URLs, prior messages).

**Enhancement**:
```python
@dataclass
class CitationContentBlock(ContentBlock):
    type: Literal["citation"] = "citation"
    source_type: Literal["url", "documentation", "prior_message", "file", "book"]
    url: str | None = None
    title: str | None = None
    author: str | None = None
    quote: str | None = None  # Cited text
    accessed_at: datetime | None = None
    confidence: float | None = None  # 0-1 if extracted via heuristic
```

**Extraction**:
- Regex for URLs: `https://...` or `docs.python.org/...`
- Markdown links: `[title](url)`
- "See also" references
- Prior conversation references ("In our earlier discussion...")

**Storage**:
```sql
CREATE TABLE citations (
    citation_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT,
    title TEXT,
    quote TEXT,
    accessed_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);

CREATE INDEX idx_citations_url ON citations(url);
```

**User Value**:
- Build personal knowledge graph of referenced materials
- Track which docs you reference most frequently
- Identify outdated references when AI knowledge cutoff updates
- Fact-check conversations by source verification

---

### 3. Mathematical Expression Block

**Files Affected**: `lib/models.py`, importers, `rendering/renderers/html.py`

**Problem**: LaTeX/math rendered as plain text, breaking search and proper rendering.

**Enhancement**:
```python
@dataclass
class MathContentBlock(ContentBlock):
    type: Literal["math"] = "math"
    format: Literal["latex", "asciimath", "mathml"] = "latex"
    inline: bool = False  # Inline vs display
    expression: str  # Raw expression
    normalized: str | None = None  # Canonicalized form for search
```

**Extraction**:
- `$...$` for inline LaTeX
- `$$...$$` for display LaTeX
- `\[...\]` for display
- Unicode math symbols (∫, ∑, √, etc.)

**Storage**:
```sql
CREATE TABLE math_blocks (
    math_block_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    format TEXT NOT NULL,
    inline BOOLEAN,
    expression TEXT NOT NULL,
    normalized TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);
```

**Rendering**:
- HTML: Use KaTeX or MathJax for client-side rendering
- Markdown: Keep as-is for downstream tools
- TUI: ASCII approximation or fallback to text

**User Value**:
- Proper mathematical rendering in web/HTML output
- Search equations by pattern (e.g., "integrals")
- Track mathematical depth of conversations

---

### 4. Structured Data Block

**Files Affected**: `lib/models.py`, importers, `storage/store.py`

**Problem**: Tables, JSON outputs, CSV data lose structure when stored as text.

**Enhancement**:
```python
@dataclass
class StructuredDataContentBlock(ContentBlock):
    type: Literal["structured_data"] = "structured_data"
    format: Literal["table", "json", "csv", "yaml", "xml"] = "json"
    raw: str  # Original representation
    parsed: dict[str, Any] | list[Any] | None = None  # Parsed structure
    schema: dict[str, str] | None = None  # Column names/types if table
```

**Extraction**:
- Markdown tables: Parse to dict with headers
- JSON blocks: Parse and validate
- CSV blocks: Detect and parse
- YAML blocks: Parse and normalize

**Storage**:
```sql
CREATE TABLE structured_data_blocks (
    data_block_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    format TEXT NOT NULL,
    raw TEXT NOT NULL,
    parsed_json TEXT,  -- JSON serialization of parsed data
    schema_json TEXT,  -- Schema info
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);
```

**User Value**:
- Export tables to actual CSV/Excel
- Re-analyze structured outputs
- Search tables by content ("find conversations with sales data")
- Preserve data integrity for re-processing

---

## Priority 2: Computed Metadata Enrichment

### 5. Topic Extraction

**Files Affected**: New `discovery/topics.py`, `lib/models.py`, `storage/store.py`

**Problem**: No topic classification. Users must remember context manually.

**Enhancement**: Add conversation-level and message-level topics:
```python
class Conversation(ConversationRecord):
    topics: list[str] | None = None  # ["python", "async", "debugging"]
    primary_topic: str | None = None
    topic_confidence: float | None = None  # 0-1 if computed

# Storage
CREATE TABLE conversation_topics (
    conversation_id TEXT PRIMARY KEY,
    topics TEXT,  -- JSON array
    primary_topic TEXT,
    topic_confidence REAL,
    computed_at TEXT,
    method TEXT,  -- "tfidf", "kmeans", "hdbscan", "llm"
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX idx_topics ON conversation_topics(primary_topic);
```

**Implementation Options**:
1. **Rule-based** (fast, cheap): Keyword extraction + frequency analysis
   - Extract 50+ frequent terms per conversation
   - Score by TF-IDF within archive
   - Threshold top 3-5 terms

2. **Embedding-based** (medium cost): Qdrant clustering + representative keywords
   - Use existing Voyage embeddings
   - HDBSCAN clustering
   - Extract top TF-IDF terms per cluster

3. **LLM-assisted** (high cost, highest quality):
   - Batch process with cheap model
   - Summarize conversation, extract topics
   - Claude 3.5 Haiku: ~$0.001 per conversation

**User Value**:
- Browse conversations by topic
- "What did I discuss about X?"
- Track expertise areas over time

---

### 6. Technical Domain Classification

**Files Affected**: `discovery/domains.py`, `lib/models.py`

**Problem**: No structured profiling by technical domain.

**Enhancement**: Multi-label domain classification:
```python
class Conversation(ConversationRecord):
    domains: dict[str, float] = {}  # {"web_frontend": 0.8, "databases": 0.3}
    primary_domain: str | None = None

# Storage
CREATE TABLE conversation_domains (
    conversation_id TEXT PRIMARY KEY,
    domains_json TEXT,  -- {"web_frontend": 0.8, ...}
    primary_domain TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX idx_primary_domain ON conversation_domains(primary_domain);
```

**Domains** (predefined, extensible):
- web_frontend, web_backend
- databases, data_engineering
- devops, infrastructure, cloud
- machine_learning, data_science
- systems, embedded, network
- security, cryptography
- testing, qa
- architecture, design_patterns

**Classification Method**:
```python
DOMAIN_KEYWORDS = {
    "web_frontend": ["react", "vue", "typescript", "css", "html", "jsx"],
    "databases": ["sql", "postgres", "mongodb", "redis", "query", "schema"],
    "devops": ["kubernetes", "docker", "terraform", "ci/cd", "deployment"],
    # ...
}

def classify_domains(conversation: Conversation) -> dict[str, float]:
    """Score conversation by domain keyword frequency."""
    text = " ".join([m.text for m in conversation.messages]).lower()
    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text)
        scores[domain] = matches / len(keywords)

    return {k: v for k, v in scores.items() if v > 0.2}  # Threshold 20%
```

**User Value**:
- Filter by specialization ("show me DevOps conversations")
- Track professional development
- Identify underexplored areas

---

### 7. Complexity Scoring

**Files Affected**: `discovery/complexity.py`, `lib/models.py`

**Problem**: Can't identify which conversations are worth revisiting or contain deep learning.

**Enhancement**: Message and conversation complexity metrics:
```python
class Message(MessageRecord):
    complexity_score: float | None  # 0-1 scale
    readability_grade: float | None  # Flesch-Kincaid
    technical_density: float | None  # Jargon ratio
    substantive_score: float | None  # Depth of reasoning

class Conversation(ConversationRecord):
    avg_complexity: float | None
    max_complexity: float | None
    complexity_trend: Literal["increasing", "constant", "decreasing"] | None
```

**Storage**:
```sql
ALTER TABLE messages ADD COLUMN complexity_score REAL;
ALTER TABLE messages ADD COLUMN readability_grade REAL;
ALTER TABLE messages ADD COLUMN technical_density REAL;

CREATE TABLE conversation_complexity (
    conversation_id TEXT PRIMARY KEY,
    avg_complexity REAL,
    max_complexity REAL,
    complexity_trend TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
```

**Computation**:
```python
def compute_complexity(message: Message) -> dict[str, float]:
    """Compute complexity metrics for a message."""
    text = message.text

    # Flesch-Kincaid grade level
    readability = compute_readability(text)

    # Technical density: jargon/specialized terms
    technical = count_technical_terms(text) / len(text.split())

    # Complexity: length + vocabulary diversity + sentence depth
    complexity = (
        (len(text) / 1000) * 0.3 +  # Length factor
        technical * 0.5 +            # Technical terms
        (len(set(text.split())) / len(text.split())) * 0.2  # Lexical diversity
    )

    return {
        "complexity_score": min(complexity, 1.0),
        "readability_grade": readability,
        "technical_density": technical,
    }
```

**User Value**:
- "Show me conversations with high technical depth"
- Identify learning opportunities
- Track communication clarity over time

---

## Priority 3: Cross-Conversation Analysis

### 8. Conversation Linking

**Files Affected**: New `discovery/linking.py`, storage schema

**Problem**: No relationships between conversations. User must manually track follow-ups.

**Enhancement**: Structured linking between conversations:
```python
@dataclass
class ConversationLink:
    link_id: str
    source_conversation_id: str
    target_conversation_id: str
    link_type: Literal["continuation", "related", "references", "supersedes", "prerequisite"]
    confidence: float  # 0-1 if inferred
    created_at: datetime
    user_created: bool  # Explicit vs automatic
```

**Storage**:
```sql
CREATE TABLE conversation_links (
    link_id TEXT PRIMARY KEY,
    source_conversation_id TEXT NOT NULL,
    target_conversation_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    confidence REAL,
    created_at TEXT,
    user_created BOOLEAN,
    FOREIGN KEY (source_conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY (target_conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX idx_links_source ON conversation_links(source_conversation_id);
CREATE INDEX idx_links_type ON conversation_links(link_type);
```

**Link Type Semantics**:
- `continuation`: Follow-up to same topic
- `related`: Different angle on same problem
- `references`: Explicitly mentions other conversation
- `supersedes`: This one replaces that one
- `prerequisite`: Should read this first

**Inference** (automatic link suggestion):
```python
def suggest_links(conversation: Conversation) -> list[tuple[ConversationId, str, float]]:
    """Suggest related conversations based on similarity."""
    # Find semantically similar conversations
    similar = self.similarity_finder.find_similar(conversation.conversation_id, limit=10)

    # Filter and suggest links
    suggestions = []
    for similar_conv, score, link_type in similar:
        if score > 0.8:
            suggestions.append((similar_conv, link_type, score))

    return suggestions
```

**User Value**:
- Build conversation narratives and learning paths
- Find follow-ups to unresolved issues
- Trace knowledge building over time

---

### 9. Knowledge Building Timeline

**Files Affected**: `discovery/knowledge_timeline.py`, `lib/repository.py`

**Problem**: Can't track how your understanding of a topic evolved.

**Enhancement**: Timeline of conversations about the same topic:
```python
class KnowledgeTopic:
    name: str  # "async Python"
    first_conversation_id: str
    last_conversation_id: str
    conversation_count: int
    earliest_date: datetime
    latest_date: datetime
    progression: Literal["beginner", "intermediate", "advanced"]
    confidence_trend: list[float]  # Progression of mastery

def build_knowledge_timeline(topic: str) -> KnowledgeTopic:
    """Build progression for a topic."""
    convs = self.repository.iter_conversations(topics=[topic])
    sorted_convs = sorted(convs, key=lambda c: c.created_at)

    # Compute progression
    complexity_scores = [compute_avg_complexity(c) for c in sorted_convs]
    progression = estimate_progression(complexity_scores)

    return KnowledgeTopic(
        name=topic,
        first_conversation_id=sorted_convs[0].conversation_id,
        last_conversation_id=sorted_convs[-1].conversation_id,
        conversation_count=len(sorted_convs),
        earliest_date=sorted_convs[0].created_at,
        latest_date=sorted_convs[-1].created_at,
        progression=progression,
        confidence_trend=complexity_scores,
    )
```

**User Value**:
- "Your Rust knowledge evolved from beginner (Jan) to intermediate (Jun)"
- Identify topics plateauing vs actively developing
- Suggest next learning steps

---

### 10. Personal Knowledge Graph

**Files Affected**: New `discovery/kg_builder.py`, `storage/knowledge_graph.py`

**Problem**: No structured representation of concepts and their relationships.

**Enhancement**: Build a graph of entities and relationships:
```python
@dataclass
class KnowledgeGraphNode:
    id: str
    name: str  # "Python", "async/await", "FastAPI"
    node_type: Literal["language", "framework", "pattern", "tool", "concept"]
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int
    source_conversations: list[str]  # Where it appears

@dataclass
class KnowledgeGraphEdge:
    source_id: str
    target_id: str
    relationship: Literal["related", "uses", "implements", "solves", "prerequisite"]
    co_occurrence_count: int  # How often appear together

# Storage
CREATE TABLE knowledge_graph_nodes (
    node_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    node_type TEXT,
    first_mentioned TEXT,
    last_mentioned TEXT,
    mention_count INTEGER
);

CREATE TABLE knowledge_graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT,
    co_occurrence_count INTEGER,
    FOREIGN KEY (source_id) REFERENCES knowledge_graph_nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES knowledge_graph_nodes(node_id)
);
```

**Entity Extraction**:
- Programming languages: Python, Rust, JavaScript, Go, etc.
- Frameworks: FastAPI, React, Django, Axum, etc.
- Libraries: NumPy, pandas, requests, tokio, etc.
- Patterns: async/await, MVC, DDD, observer pattern, etc.
- Concepts: recursion, concurrency, abstraction, etc.
- Tools: Docker, Kubernetes, Git, PostgreSQL, etc.

**Relationship Inference**:
- Co-occurrence in same conversation → "related"
- Explicit mentions ("FastAPI uses Starlette") → "uses"
- Tool-use comments ("I used X to solve Y") → "solves"

**User Value**:
- Visual knowledge graph: see your expertise landscape
- Find knowledge gaps and prerequisites
- Identify underexplored combinations

---

## Priority 4: Enhanced Semantic Properties

### 11. Tool Use Analysis

**Files Affected**: `lib/models.py`, new `discovery/tool_analysis.py`

**Enhancement**: Richer classification of tool usage:
```python
@dataclass
class ToolUsagePattern:
    tool_name: str  # "code_interpreter", "web_search", "calculator"
    conversation_count: int
    success_rate: float  # Inferred from outcome messages
    avg_turns_to_success: int
    domains: list[str]  # Domains where used most
    first_used: datetime
    last_used: datetime
```

**Analysis**:
- Aggregate tool usage by provider
- Success rate estimates (user confirms vs error cycles)
- Correlation between tools and conversation length
- Most-used tools by domain

---

### 12. Conversation Outcomes

**Files Affected**: `lib/models.py`, new `discovery/outcomes.py`

**Problem**: No structured understanding of how conversations ended.

**Enhancement**:
```python
@dataclass
class ConversationOutcome:
    resolution: Literal["solved", "clarified", "abandoned", "ongoing"]
    resolution_confidence: float
    resolution_timestamp: datetime | None
    resolution_message_id: str | None  # Where resolution occurred
    user_satisfaction: float | None  # If manually set
    follow_up_conversation_id: str | None  # Continuation
```

**Heuristics**:
- User thanks AI → likely "solved"
- Conversation hasn't continued in 3 months → "abandoned"
- No follow-up question → "clarified"
- Continues monthly → "ongoing"

**User Value**:
- Identify unresolved issues for follow-up
- Track which conversation types resolve most effectively
- Correlate outcomes with providers

---

### 13. Code Quality Metrics

**Files Affected**: `discovery/code_analysis.py`, code block extraction

**Enhancement**: Analyze code snippets shared in conversations:
```python
@dataclass
class CodeMetrics:
    language: str
    snippet_count: int
    avg_complexity: float  # Cyclomatic complexity estimate
    uses_tested_patterns: bool
    anti_patterns_detected: list[str]
    style_issues: list[str]
```

**Analysis**:
- Detect code anti-patterns ("return True if x else False")
- Style consistency checks
- Complexity estimation
- Reference to best practices

---

## Implementation Priorities

### Phase 1 (Weeks 1-2): Content Block Extensions
- Code blocks (High ROI)
- Citations (Medium ROI)
- Structured data (Medium ROI)

**Effort**: 40 hours

### Phase 2 (Weeks 3-4): Computed Metadata
- Topic extraction (High ROI)
- Domain classification (Medium ROI)
- Complexity scoring (Medium ROI)

**Effort**: 50 hours

### Phase 3 (Weeks 5-6): Cross-Conversation Analysis
- Conversation linking (Medium ROI)
- Knowledge timeline (Medium ROI)
- Personal knowledge graph (High ROI)

**Effort**: 60 hours

### Phase 4 (Weeks 7+): Polish & Integration
- Tool use analysis
- Conversation outcomes
- Code quality metrics

**Effort**: 30 hours

---

## Success Criteria

| Feature | Success Metric | Target |
|---------|---|---|
| Content blocks | 95%+ code/citation extraction accuracy | Week 2 |
| Topic extraction | Coherent topics for 50+ conversations | Week 4 |
| Domain classification | Correct primary domain for 80%+ convs | Week 4 |
| Complexity scoring | Correlation 0.8+ with manual assessment | Week 4 |
| Conversation links | Suggest meaningful links 80%+ of time | Week 6 |
| Knowledge graph | >50 nodes for 100+ conversation archive | Week 6 |
| Outcomes | Detect resolution with 70%+ accuracy | Week 8 |

---

## Storage Strategy

- Core tables: `conversations`, `messages`, `attachments`
- Metadata tables: `message_topics`, `conversation_domains`, `conversation_complexity`
- Content block tables: `code_blocks`, `citations`, `math_blocks`, `structured_data_blocks`
- Relationship tables: `conversation_links`, `knowledge_graph_nodes/edges`
- Derived tables: `conversation_outcomes`, `tool_usage_stats`

**Total Storage**: +5-10 MB for 1000-conversation archive (mainly indexing)

---

## Backward Compatibility

All enhancements are additive:
- New columns are optional (nullable)
- Existing content_blocks preserved
- Default values for computed fields
- Migration scripts for existing archives
