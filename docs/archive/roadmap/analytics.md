# Polylogue Analytics & Insights Roadmap

## Overview

This roadmap outlines 15 analytics features to unlock deep insights about conversation patterns, learning trajectories, and AI usage. Current `view -p stats` provides basic metrics; these enhancements enable sophisticated analysis of prompting sophistication, topic evolution, provider effectiveness, and personal knowledge development.

**Current State**:
- Basic stats projection (conversation count, message count)
- No temporal analysis
- No topic tracking
- No complexity metrics
- No comparative analytics (provider, domain)

---

## Tier 1: High-Impact, Immediate "Aha Moments"

### 1. Conversation Complexity Timeline

**Files Affected**: New `analytics/complexity.py`, `cli/commands/analyze.py`

**Insight**: How your prompting sophistication evolves over time.

**Data Sources**:
- Word count per message (already computed in `Message.word_count`)
- `is_tool_use`, `is_thinking` classification
- Turn count (message pairs)
- Timestamps

**Metrics**:
```python
@dataclass
class ComplexityMetrics:
    period: str  # "week", "month", "year"
    avg_user_prompt_length: int  # Words
    avg_assistant_response_length: int
    tool_use_frequency: float  # Ratio
    thinking_trace_frequency: float  # Ratio
    turns_per_conversation: float  # Average
    complexity_score: float  # 0-1 composite
```

**Visualization**:
```bash
polylogue analyze --complexity-timeline

# Output:
# Prompting Complexity Evolution
# ═════════════════════════════════════
#
# User Prompt Length (words)
#
#     400 │     ╱─╲
#     350 │    ╱   ╲─╱─╲
#     300 │   ╱         ╲─╱─
#     250 │──╱
#         │
#        Jan Feb Mar Apr May Jun
#
# Tool Use Frequency: ▓▒▒░░░░░░░ (10% → 18%)
# Thinking Traces: ░░░░░░░░░░ (0% → 3%)
# Conversation Turns: 15 → 22
#
# Insight: Your prompting sophistication increased 35% over 6 months.
# Recommendation: Review foundational concepts to ensure depth.
```

**Implementation**:
```python
# analytics/complexity.py
def compute_complexity_timeline(
    repository: StorageRepository,
    period: Literal["week", "month", "year"] = "month"
) -> dict[str, ComplexityMetrics]:
    """Compute complexity metrics per time period."""
    convs = list(repository.iter_conversations())

    # Group by period
    by_period = defaultdict(list)
    for conv in convs:
        key = conv.created_at.strftime(PERIOD_FORMAT[period])
        by_period[key].append(conv)

    # Compute metrics
    results = {}
    for period_key, period_convs in sorted(by_period.items()):
        user_messages = [m for c in period_convs for m in c.messages if m.role == "user"]
        assistant_messages = [m for c in period_convs for m in c.messages if m.role == "assistant"]

        results[period_key] = ComplexityMetrics(
            period=period_key,
            avg_user_prompt_length=np.mean([m.word_count for m in user_messages]),
            avg_assistant_response_length=np.mean([m.word_count for m in assistant_messages]),
            tool_use_frequency=sum(1 for m in assistant_messages if m.is_tool_use) / len(assistant_messages),
            thinking_trace_frequency=sum(1 for m in assistant_messages if m.is_thinking) / len(assistant_messages),
            turns_per_conversation=np.mean([len(c.messages) for c in period_convs]) / 2,
        )

    return results
```

**Success Criteria**:
- Detect 20%+ changes in complexity
- Identify trend reversals (plateaus)
- Provide actionable recommendations

---

### 2. Topic Discovery & Clustering

**Files Affected**: New `analytics/topic_clustering.py`

**Insight**: "You've discussed Python 234 times, but only 12 times about testing."

**Approach**:
- TF-IDF term extraction from conversations
- K-means clustering on Qdrant embeddings (if available)
- Topic label assignment via representative terms

**Metrics**:
```python
@dataclass
class Topic:
    name: str  # "python", "async", "web"
    conversation_count: int
    message_count: int
    frequency_rank: int  # Most to least discussed
    first_mentioned: datetime
    last_mentioned: datetime
    trend: Literal["increasing", "stable", "decreasing"]
```

**Visualization**:
```bash
polylogue analyze --topics

# Output:
# Top Topics by Conversation Count
# ════════════════════════════════════
#
# 1. Python        ████████████ (234 convs, 3,421 messages)
#    └─ Subtopics: async, testing, decorators, debugging
#    └─ Trend: stable (monthly avg 18.2)
#
# 2. Web           ████████ (156 convs, 2,101 messages)
#    └─ Subtopics: react, typescript, api, css
#    └─ Trend: increasing (+3.2% MoM)
#
# 3. DevOps        ████░░░░░░░░ (87 convs, 1,234 messages)
#    └─ Subtopics: docker, kubernetes, terraform
#    └─ Trend: decreasing (-2.1% MoM)
#
# 4. Rust          ████░░░░░░░░ (81 convs, 1,098 messages)
#    └─ Subtopics: async, ownership, traits, wasm
#    └─ Trend: increasing (+5.3% MoM)
#
# 5. Testing       ██░░░░░░░░░░ (34 convs, 421 messages)
#    └─ Subtopics: pytest, mocking, integration tests
#    └─ Trend: decreasing (-1.2% MoM)
#
# Insight: You're spending more time on Rust and Web,
# less on DevOps. Testing volume declining (concerning).
```

**Implementation**:
```python
# analytics/topic_clustering.py
def extract_topics(
    repository: StorageRepository,
    n_topics: int = 10
) -> dict[str, Topic]:
    """Extract main topics from conversations."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Collect conversation texts
    texts = []
    for conv in repository.iter_conversations():
        text = " ".join([m.text for m in conv.messages])
        texts.append(text)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        max_df=0.8,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Extract top terms
    terms = vectorizer.get_feature_names_out()
    topics = {}

    for i in range(n_topics):
        top_indices = tfidf_matrix[:, i].toarray().flatten().argsort()[-10:]
        topic_terms = [terms[idx] for idx in top_indices]
        topic_name = topic_terms[0]  # Primary term

        topics[topic_name] = Topic(
            name=topic_name,
            conversation_count=sum(1 for text in texts if topic_name in text.lower()),
            # ... other metrics
        )

    return topics
```

**Success Criteria**:
- Extract 10-30 coherent topics
- Topic names meaningful and interpretable

---

### 3. Provider Comparison Dashboard

**Files Affected**: New `analytics/provider_comparison.py`

**Insight**: Which provider gives better responses for what tasks?

**Metrics**:
```
         Claude  ChatGPT  Gemini
─────────────────────────────────
Avg Response Length    450 words  320 words  380 words
Response Time          2.3s       1.8s       1.5s
Tool Use Frequency     18%        5%         2%
Thinking Trace %       12%        0%         0%
Avg Cost/Conversation  $0.15      N/A        N/A
Conversation Length    22 turns   16 turns   8 turns
```

**Implementation**:
```python
# analytics/provider_comparison.py
def compare_providers(repository: StorageRepository) -> ProviderComparison:
    """Compare metrics across providers."""
    stats_by_provider = {}

    for provider_name in set(c.provider_name for c in repository.iter_conversations()):
        convs = list(repository.iter_conversations(provider_name=provider_name))

        messages = []
        for conv in convs:
            messages.extend(conv.messages)

        assistant_msgs = [m for m in messages if m.role == "assistant"]

        stats_by_provider[provider_name] = ProviderStats(
            conversation_count=len(convs),
            avg_response_length=np.mean([m.word_count for m in assistant_msgs]),
            tool_use_frequency=sum(1 for m in assistant_msgs if m.is_tool_use) / len(assistant_msgs),
            thinking_trace_frequency=sum(1 for m in assistant_msgs if m.is_thinking) / len(assistant_msgs),
            avg_conversation_length=np.mean([len(c.messages) for c in convs]),
        )

    return ProviderComparison(stats=stats_by_provider)
```

**Success Criteria**:
- Compare 3+ providers accurately
- Highlight strongest/weakest areas per provider

---

### 4. Learning Trajectory Analyzer

**Files Affected**: New `analytics/learning_trajectory.py`

**Insight**: "Your Python questions evolved from basic syntax to async patterns."

**Approach**:
- Extract keywords from user questions over time
- Score complexity using domain-specific heuristics
- Build trajectory from beginner→advanced

**Output**:
```bash
polylogue analyze --learning-trajectory python

# Output:
# Python Learning Trajectory (Jan 2024 - Jun 2025)
# ═════════════════════════════════════════════════
#
# Phase 1: Basics (Jan-Feb)
#   - Keywords: print, variable, loop, function
#   - Sample Q: "How do I iterate over a list?"
#   - Complexity: ★☆☆
#
# Phase 2: Intermediate (Mar-Apr)
#   - Keywords: decorator, comprehension, context_manager, async
#   - Sample Q: "How do I write a decorator?"
#   - Complexity: ★★☆
#
# Phase 3: Advanced (May-Jun)
#   - Keywords: asyncio, event_loop, green_threads, metaclass
#   - Sample Q: "Compare asyncio vs trio for concurrent I/O"
#   - Complexity: ★★★
#
# Trajectory: Linear progression, consistent learning rate
# Next: Explore functional programming (none so far)
```

**Success Criteria**:
- Identify 3-5 learning phases
- Detect plateaus or reversals

---

### 5. Conversation Network Graph

**Files Affected**: New `analytics/network_graph.py`, export formats

**Insight**: How topics connect across conversations.

**Visualization**:
- Nodes: Conversations (sized by message count)
- Edges: Semantic similarity >threshold
- Clusters: Topic domains (color-coded)

**Implementation**:
```python
# analytics/network_graph.py
def build_conversation_network(
    repository: StorageRepository,
    min_similarity: float = 0.75
) -> nx.DiGraph:
    """Build conversation similarity graph."""
    convs = list(repository.iter_conversations())
    graph = nx.DiGraph()

    # Add nodes
    for conv in convs:
        graph.add_node(
            conv.conversation_id,
            title=conv.title,
            message_count=len(conv.messages),
            provider=conv.provider_name
        )

    # Add edges (similarity)
    for i, conv1 in enumerate(convs):
        for conv2 in convs[i+1:]:
            similarity = compute_similarity(conv1, conv2)
            if similarity > min_similarity:
                graph.add_edge(conv1.conversation_id, conv2.conversation_id, weight=similarity)

    return graph

# Export to JSON for D3.js or Gephi
def export_network(graph: nx.DiGraph, format: str = "json") -> str:
    """Export graph for visualization."""
    nodes = [{"id": n, **graph.nodes[n]} for n in graph.nodes()]
    edges = [{"source": u, "target": v, **graph[u][v]} for u, v in graph.edges()]
    return json.dumps({"nodes": nodes, "edges": edges})
```

**CLI**:
```bash
polylogue analyze --network --output network.json
# Then visualize with:
# npx force-graph --input network.json
```

**Success Criteria**:
- Build graph for 100+ conversations: <5 seconds
- Export to multiple formats (JSON, GraphML, etc.)

---

## Tier 2: Deeper Insights with Moderate Effort

### 6. Temporal Activity Heatmap

**Files Affected**: New `analytics/temporal.py`, `cli/commands/heatmap.py`

**Insight**: When you chat with AI (hour of day, day of week patterns).

**Visualization**:
```bash
polylogue analyze --heatmap activity

# Hour of Day Heatmap
# ───────────────────
#     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
# Mon  .   .   .   .   .   .   1   3   5  12  18  15  14   8   5   4   3   2   1   .   .   .   .   .
# Tue  .   .   .   .   .   .   2   4   8  14  20  16  12  10   6   5   4   2   1   .   .   .   .   .
# Wed  .   1   .   .   .   1   3   5   9  16  19  18  13   9   7   6   5   3   1   1   .   .   .   .
# Thu  .   .   .   .   .   .   2   5  10  15  18  14  11   8   5   4   3   2   1   .   .   .   .   .
# Fri  .   .   .   .   .   .   1   3   6  12  16  14  10   7   5   4   3   2   1   .   .   .   .   .
# Sat  .   .   1   .   .   .   .   1   3   8  12  10   9   5   4   3   2   2   1   1   .   .   .   .
# Sun  .   .   .   .   .   .   .   2   4   9  14  12   8   6   4   3   2   1   1   .   .   .   .   .
#
# Peak hours: 10:00-11:00 (48 messages)
# Least active: 0:00-6:00, 23:00
# Most active day: Wednesday (92 messages)
```

**Implementation**:
```python
# analytics/temporal.py
def compute_activity_heatmap(repository: StorageRepository) -> dict[str, dict[int, int]]:
    """Compute activity by hour and weekday."""
    heatmap = {day: defaultdict(int) for day in range(7)}

    for message in repository.iter_messages():
        hour = message.timestamp.hour
        weekday = message.timestamp.weekday()
        heatmap[weekday][hour] += 1

    return heatmap
```

**Success Criteria**:
- Accurate hour/weekday attribution
- Identify circadian patterns

---

### 7. Conversation Depth Analyzer

**Files Affected**: `analytics/depth.py`

**Insight**: Distribution of conversation lengths (quick Q&A vs deep dives).

**Metrics**:
```
Conversation Length Distribution
═════════════════════════════════

Message Count
├─ 1-5 messages:      ████░░░░░░░░ (25%, 312 conversations)
│  └─ Quick questions
├─ 6-20 messages:     ████████░░░░ (40%, 501 conversations)
│  └─ Standard interactions
├─ 21-50 messages:    ░░░░░░░░░░░░ (23%, 287 conversations)
│  └─ Deep dives
└─ 50+ messages:      ░░░░░░░░░░░░ (12%, 150 conversations)
   └─ Marathon sessions

Deep Dive Threshold: 20 turns, >5000 words
- Count: 437 conversations (26%)
- Avg word count: 8,234
- Avg duration: 3.2 hours (estimated)

Insight: Most conversations are brief. Deep dives concentrated
in Python and debugging topics.
```

---

### 8. Question Type Taxonomy

**Files Affected**: New `analytics/question_types.py`

**Insight**: What kinds of questions you ask most.

**Classification**:
```python
QUESTION_PATTERNS = {
    "how_to": r"^(how (do|can|to)|steps to|guide to|tutorial)",
    "why": r"^(why|what's the reason|explain why)",
    "debug": r"(error|exception|bug|issue|problem|not working|crash)",
    "design": r"(architecture|design|pattern|structure|should i)",
    "compare": r"(vs|difference between|compare|which one)",
    "generate": r"(write|create|generate|code for|make)",
}
```

**Output**:
```bash
polylogue analyze --question-types

# Question Type Distribution
# ═════════════════════════════
#
# How-To        ███████░░░░░░░░░░ (35%, 432 questions)
# Design        █████░░░░░░░░░░░░ (25%, 308 questions)
# Debug         ████░░░░░░░░░░░░░ (20%, 246 questions)
# Compare       ███░░░░░░░░░░░░░░ (15%, 184 questions)
# Generate      ██░░░░░░░░░░░░░░░ (5%, 61 questions)
#
# Insight: You ask more design/architecture questions than debugging.
# Trend: Compare questions increased 50% (more evaluative).
```

---

### 9. Response Quality Signals

**Files Affected**: New `analytics/response_quality.py`

**Insight**: Which conversations had friction vs smooth exchanges.

**Proxy Signals**:
- Multiple clarification rounds
- Conversation length without apparent resolution
- Thinking trace frequency (indicates complexity)
- Tool use patterns (successful vs failed attempts)

**Heuristics**:
```python
class ResponseQualityScore:
    def __init__(self, conversation: Conversation):
        self.conversation = conversation

    def compute(self) -> tuple[float, dict[str, float]]:
        """Compute quality score 0-1."""
        scores = {
            "resolution": self._score_resolution(),
            "efficiency": self._score_efficiency(),
            "engagement": self._score_engagement(),
            "depth": self._score_depth(),
        }

        overall = np.mean(list(scores.values()))
        return overall, scores

    def _score_resolution(self) -> float:
        """Did conversation reach resolution?"""
        # Look for thank you, confirmation, etc.
        last_messages = [m.text for m in self.conversation.messages[-5:]]
        resolution_keywords = ["thank", "perfect", "got it", "yes that works"]
        matches = sum(1 for m in last_messages for kw in resolution_keywords if kw in m.lower())
        return min(matches / 5, 1.0)
```

---

### 10. Personal Knowledge Graph

**Files Affected**: New `analytics/knowledge_graph.py`, export formats

**Insight**: A graph of concepts/entities you've discussed.

**Entities**:
- Languages (Python, Rust, Go, JavaScript)
- Frameworks (FastAPI, React, Django, Axum)
- Libraries (NumPy, pandas, tokio, etc.)
- Patterns (async/await, MVC, DDD)
- Concepts (recursion, concurrency, abstraction)

**Graph Structure**:
- Nodes: Entities + frequency + first/last mention
- Edges: Co-occurrence relationships
- Attributes: Confidence, evidence count

**Implementation**:
```python
# analytics/knowledge_graph.py
def build_knowledge_graph(repository: StorageRepository) -> nx.DiGraph:
    """Build graph of entities and relationships."""
    graph = nx.DiGraph()

    # Extract entities from all messages
    for message in repository.iter_messages():
        entities = extract_entities(message.text)
        for entity in entities:
            graph.add_node(entity.name, type=entity.type, first_seen=entity.timestamp)

        # Add co-occurrence edges
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                if graph.has_edge(ent1.name, ent2.name):
                    graph[ent1.name][ent2.name]['weight'] += 1
                else:
                    graph.add_edge(ent1.name, ent2.name, weight=1)

    return graph
```

**Export to JSON-LD** or **RDF** for external tools:
```bash
polylogue analyze --knowledge-graph --format jsonld > kg.json
# Import into: neo4j, ontodia, knowledge graph visualizers
```

**Success Criteria**:
- Extract 50-100+ entities
- Meaningful entity relationships
- Export to visualization tools

---

### 11. Tool Use Analysis

**Files Affected**: New `analytics/tool_usage.py`

**Insight**: How effectively you use AI tools.

**Metrics**:
```python
@dataclass
class ToolUsageStats:
    tool_name: str
    usage_count: int
    success_rate: float  # Inferred from outcomes
    avg_turns_to_success: int
    domains: list[str]  # Where used most
    trend: Literal["increasing", "stable", "decreasing"]
```

**Analysis**:
```bash
polylogue analyze --tool-usage

# Tool Usage Analysis
# ═════════════════════
#
# code_interpreter (ChatGPT)
#   - Usage: 87 times
#   - Success rate: 78%
#   - Avg turns to success: 2.1
#   - Domains: data_analysis (45%), web_dev (32%)
#   - Trend: increasing (+5% MoM)
#
# web_search (ChatGPT)
#   - Usage: 23 times
#   - Success rate: 95%
#   - Avg turns to success: 1.0
#   - Domains: research (60%), current_events (40%)
#   - Trend: stable
#
# claude_code_tools (Claude)
#   - Usage: 156 times
#   - Success rate: 92%
#   - Avg turns to success: 1.3
#   - Domains: development (100%)
#   - Trend: increasing (+12% MoM)
```

---

### 12. Conversation Outcomes

**Files Affected**: New `analytics/outcomes.py`

**Insight**: How conversations end and whether they resolve.

**Classification**:
```python
class ConversationOutcome(Enum):
    SOLVED = "solved"  # Problem resolved
    CLARIFIED = "clarified"  # Understanding achieved
    ABANDONED = "abandoned"  # No follow-up
    ONGOING = "ongoing"  # Still active
    DELEGATED = "delegated"  # Plan to implement later
```

**Heuristics**:
```python
def classify_outcome(conversation: Conversation) -> ConversationOutcome:
    """Infer outcome from conversation patterns."""
    # Recent activity check
    age_days = (datetime.now() - conversation.updated_at).days

    if age_days > 90:
        if len(conversation.messages) < 5:
            return ConversationOutcome.ABANDONED

    # Look for thank you/resolution signals
    final_messages = conversation.messages[-3:]
    resolution_signals = ["thank", "perfect", "got it", "working now", "solves", "brilliant"]
    if any(sig in " ".join(m.text.lower() for m in final_messages) for sig in resolution_signals):
        return ConversationOutcome.SOLVED

    # Look for implementation deferral
    deferral_signals = ["try it", "i'll implement", "implement this", "later"]
    if any(sig in " ".join(m.text.lower() for m in final_messages) for sig in deferral_signals):
        return ConversationOutcome.DELEGATED

    # Still getting updates
    if age_days < 7:
        return ConversationOutcome.ONGOING

    return ConversationOutcome.CLARIFIED
```

---

## Tier 3: Strategic Insights (Higher Complexity)

### 13. Prompting Technique Analysis

**Files Affected**: New `analytics/prompting_techniques.py`

**Insight**: What prompting techniques work best for you.

**Techniques**:
- Chain-of-thought (explicit reasoning)
- Few-shot examples
- System prompts/role-playing
- Decomposition (break into steps)
- Direct vs exploratory

**Analysis**:
```bash
polylogue analyze --prompting-techniques

# Prompting Technique Effectiveness
# ══════════════════════════════════
#
# Chain-of-Thought
#   - Usage: 156 conversations (12%)
#   - Avg resolution turns: 1.8
#   - Success rate: 88%
#
# Few-Shot Examples
#   - Usage: 89 conversations (7%)
#   - Avg resolution turns: 1.2 (fastest!)
#   - Success rate: 92%
#
# Role-Playing/System Prompt
#   - Usage: 45 conversations (3%)
#   - Avg resolution turns: 2.4
#   - Success rate: 76%
#
# Insight: Few-shot examples are most effective for you.
# Consider using more in future conversations.
```

---

### 14. Cost & Efficiency Analysis

**Files Affected**: New `analytics/cost_efficiency.py` (if cost data available)

**Insight**: Cost per conversation, cost per solution, efficiency trends.

**Metrics** (where available from provider APIs):
- Cost per conversation
- Cost per solved problem
- Tokens used (efficiency)
- Provider cost comparison

---

### 15. Expertise & Knowledge Gaps

**Files Affected**: New `analytics/expertise.py`

**Insight**: Identify skill levels and knowledge gaps.

**Analysis**:
```bash
polylogue analyze --expertise

# Expertise Profile
# ═════════════════
#
# Advanced (>50 conversations)
#   - Python (234 convs)
#   - Web Development (156 convs)
#   - DevOps (87 convs)
#
# Intermediate (20-50 conversations)
#   - Rust (81 convs)
#   - Machine Learning (45 convs)
#   - Mobile Development (32 convs)
#
# Beginner (5-20 conversations)
#   - C++ (18 convs)
#   - Kubernetes (14 convs)
#   - GraphQL (9 convs)
#
# Never Discussed
#   - Elixir
#   - Erlang
#   - Haskell
#   - Clojure
#
# Recommendation: Deepen Rust knowledge (primed for intermediate→advanced).
# Gap: Testing practices across all domains (only 12 conversations).
```

---

## Implementation Priorities

### Phase 1 (Weeks 1-2): Immediate Insights
- Complexity timeline (High impact)
- Topic discovery (High impact)
- Provider comparison (Medium impact)

**Effort**: 30 hours

### Phase 2 (Weeks 3-4): Temporal Analysis
- Activity heatmap (Medium impact)
- Conversation depth (Medium impact)
- Outcome classification (Medium impact)

**Effort**: 25 hours

### Phase 3 (Weeks 5-6): Advanced Analysis
- Learning trajectory (High impact)
- Conversation network (High impact)
- Question type taxonomy (Medium impact)

**Effort**: 40 hours

### Phase 4 (Weeks 7-8): Knowledge Graphs
- Personal knowledge graph (High impact)
- Entity relationships (Medium impact)
- Expertise profiling (Medium impact)

**Effort**: 35 hours

---

## CLI Interface

```bash
# Analytics command tree
polylogue analyze
├── --complexity-timeline          # Prompting sophistication evolution
├── --topics                       # Topic discovery & clustering
├── --provider-comparison          # Compare providers
├── --learning-trajectory <topic>  # Learning progression
├── --network                      # Conversation similarity graph
├── --heatmap <type>              # Activity heatmaps
├── --depth                        # Conversation length distribution
├── --question-types              # Question classification
├── --response-quality            # Quality signals
├── --knowledge-graph             # Entity relationships
├── --tool-usage                  # Tool effectiveness
├── --outcomes                    # Conversation resolution
├── --prompting-techniques        # Technique effectiveness
├── --expertise                   # Skill profiling
└── --report                      # Comprehensive report (all above)

Options:
  --output <format>               # json, csv, html, pdf
  --since <date>                  # Date filter
  --until <date>
  --provider <name>               # Filter by provider
  --topic <name>                  # Filter by topic
  --export <path>                 # Save for external tools
```

---

## Success Metrics

- **Discoverability**: Users find at least 3 insights per analysis run
- **Accuracy**: Metric correlations match manual review (>0.85)
- **Performance**: Analysis of 1000+ conversations: <30 seconds
- **Actionability**: 80%+ of insights lead to behavior changes
- **Engagement**: 60%+ of users run analysis at least monthly

---

## Data Requirements

All analytics use existing data or annotations:
- Message timestamps (already stored)
- Message text and role (already stored)
- Content blocks (already extracted)
- Conversation metadata (already stored)
- Provider name (already tracked)

No additional data collection needed.

---

## Output Formats

| Feature | Formats |
|---------|---------|
| Heatmaps | ASCII, JSON, CSV, PNG/PDF (via matplotlib) |
| Graphs | JSON (D3.js), GraphML (Gephi), JSON-LD, RDF |
| Reports | HTML, PDF, Markdown, JSON |
| Timelines | ASCII, JSON, CSV, SVG |
| Comparisons | Tables (ASCII, CSV), JSON, HTML |

---

## Roadmap Dependencies

- Phase 1: No external dependencies
- Phase 2: datetime parsing (stdlib)
- Phase 3: NumPy, scikit-learn (optional, degrade gracefully)
- Phase 4: NetworkX (optional), NER library (optional)

All analytics work with core Polylogue data; advanced features gracefully degrade if optional deps unavailable.
