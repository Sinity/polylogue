# Uses the operator imagined, from their own chats and notes

What the operator wanted to do with the exocortex, gathered from their own
words: the uses, questions, routines and artifacts that should grow on
Polylogue and Sinex. Seed material for the vision catalogs
(`polylogue-spec/vision/`, `sinex-spec/vision/`, and the joint catalog
here); origins of the designs are in the intent files, not repeated here.

**Corpus.** The pre-reset Polylogue index
(`/realm/state/polylogue-aside-20260926T182000Z/.index-generations/gen-1785377665711-06297b00/index.db`,
read in place with `immutable=1`): ChatGPT 2,652, claude.ai 431, AI Studio
239 and Grok 6 web threads (December 2022 to July 2026), searched by
concept, not by project name, with the operator's own turns as the signal.
Notes at lower weight: `/realm/personal/idea/`, `/realm/personal/goal/`,
`/realm/project/knowledge-system/note/`.

**Form.** Each use: what the operator wanted (one line), short dated quotes
(verbatim, typos kept), and the tag `polylogue`, `sinex` or `both`. Personal,
health and financial content is left out unless it states a use, and then
only as the use, never the details.

**Split.** Two sessions write this file, each only under its own heading.

## Conversation, agent and context uses (polylogue session)

Searched 2026-10-12 over the operator's own text turns in the web origins
(threads over ~6,000 characters per turn skipped as pastes). AI Studio turns
carry no time; they are marked "AI Studio, undated". Coverage pointers are to
`polylogue-spec/` at this date. Uses the Sinex section already holds are
cross-referenced, not repeated.

### Getting the value out of conversations

- **One long chat distilled into a standalone artifact** (`polylogue`;
  `distill-thread`). AI Studio, 2025-06-01: "I want you to thoroughly unify
  all value and detail out of the attached chatlog." AI Studio, undated: "a
  standalone artifact which encompasses >85% (target 98%) of info/value from
  the whole chatlog". claude.ai, 2026-06-27: "full standalone writeup,
  containing everything of value during this chatlog, deduped, ordered
  reasonably". The most repeated request in the corpus. Spec: partly
  (`spec/workflows.md` §10 reads a long session quickly); a kept distillate
  tied to its source session, re-made when the session grows, is not
  covered.
- **Several branches or copies of one chat brought together** (`polylogue`;
  `merge-branches`). ChatGPT, 2026-06-01: "Other ones are actually other
  branches of this chat, whiich I wish to bring together right here."
  ChatGPT, 2026-06-13: "you've got multiple copies of the chatlogs, not sure
  how they differ". Spec: copies of one conversation merge
  (`decisions/merging.md`); a view that lines up sibling branches and shows
  what each added is not covered.
- **Never missing what an old chat held, without rereading it**
  (`polylogue`; `ask-not-reread`). AI Studio, undated: "I have thhe
  discomfort that I'm missing something valuable from earlier on, and feel
  the need to \"go thoruhg things\" but it is aversive"; "Instead, I could
  just ask". Spec: covered by search, recall and model-read answers
  (`spec/workflows.md` §3 and §12; `spec/user-data.md`, Recall;
  `spec/analysis.md` on meaning).
- **Chats checked against the current state of a project** (`both`;
  `chats-vs-project`). ChatGPT, 2026-06-16: "really deeply analyze the
  chatlogs against current cod[e]". ChatGPT, 2026-06-19: "outline of still
  applicable/relevant content from this chatlog, given current state of the
  project". ChatGPT, 2026-06-23: "unrealized ideas, opportunities?" Spec:
  not covered (git correlation, `spec/analysis.md` §9, links commits, not
  ideas).
- **All my ideas, and how they tie together** (`both`; `idea-ledger`;
  related to the Sinex section's `project-ledger`). ChatGPT, 2025-05-10: "an
  overview which would let me understand how all of my ideas for what to do
  tie together. Now I'm kinda adrift". AI Studio, 2025-08-28: "my ideas are
  not even existing projects, but existing _concepts_." Spec: not covered
  for ideas raised in chats.
- **Notes written from chats** (`both`; `chats-to-notes`). AI Studio,
  2025-04-24 (a journal line quoted in a thread): "primarily AI-written
  Vault, or partially, might actually be a good idea if handled well. This
  would serve as perfect way to capture / process value out of AI
  chatlogs". AI Studio, 2025-04-08: a "chatlog to obsidian markdown script".
  The operator's notes already cite chats as sources in their front matter
  (`extrinsic.chatlog.gemini.*` in `genesis.md` and others). Spec: partly
  (Markdown export, `spec/export-format.md`; notes, `spec/user-data.md`); a
  two-way link between a vault note and the session it came from is not
  covered.
- **A map of my stated views, checked for contradictions** (`both`;
  `stance-map`). ChatGPT, 2025-04-25: "produce a clear, contradiction-checked
  map of my stances". AI Studio, undated: "potential contradictions between
  my stated beliefs and ones impliled by the analysis". ChatGPT, 2026-06-05:
  "self-contained model of myself, my goals etc." Spec: not covered.
- **Replies drafted in my own voice, from my own writing** (`both`;
  `my-voice`). ChatGPT, 2026-04-09: "I want the reply to be in my voice,
  stylistically and also semantically (I mean, grok my POV / worldview
  etc.)", with an archive of the operator's comments attached. Spec: not
  covered.
- **Learning by prompts, not flashcards** (`both`; `prompted-review`).
  ChatGPT, 2025-04-26: "custom spaced repetition, if at all. I think it might
  be less flashcards and more... prompts". ChatGPT, 2025-09-18: "How can one
  use ChatGPT to efficiently work together with some sort of spaced
  repetition software?" Spec: not covered.
- **Long-lived chats per topic, not one-off threads** (`polylogue`;
  `topic-threads`). AI Studio, 2025-09-19: "these should actually be perma ai
  chats, maybe using 'projects' feeature, not just notes." Spec: partly
  (pages and investigations, later, `spec/user-data.md`).
- **A conversation as a tree with named branches** (`polylogue`;
  `branch-tree`). ChatGPT, 2023-07-08, designing an LLM CLI: "It should be
  like a tree - conversation could branch."; "Shouldn't branches have
  names?" Spec: branches are modelled and rendered (`spec/model.md`,
  `spec/rendering.md`); names for branches are not.

### Finding things across the archive

- **A half-remembered chat found again** (`polylogue`; `fuzzy-recall`).
  claude.ai, 2025-11-13: a conversation "which started as translation if I
  remember correctly ... can you perhaps unearth it?" claude.ai, 2025-12-06:
  "find chats which seem at all relevant to the following tweet ... IDK how
  to find it, looking through 800+ chatlogs". Spec: covered by full-text and
  semantic search (`spec/query.md`; `spec/analysis.md` §8, slice 9).
- **What a model told me, quoted back** (`polylogue`; `quote-back`).
  ChatGPT, 2026-06-18: "do tell me everything you told me about the
  situation, quote basically." Spec: covered (`spec/workflows.md` §3).
- **An analysis reused instead of redone in another chat** (`polylogue`;
  `kept-analysis`). ChatGPT, 2025-04-17: "can avoid analysis work
  duplicating when I decide to analyze this in some other chat". Spec:
  covered (saved queries and kept results, `spec/user-data.md`).
- **Behaviour of models studied across many chats** (`polylogue`;
  `model-behaviour-study`). claude.ai, 2025-12-06 (above): a peculiar
  pattern where a model "seemed to \"fight itself\"". Spec: partly
  (`spec/workflows.md` §21); a rubric pass that labels such moments across
  the archive fits `spec/user-data.md`, annotation passes.
- **My AI chats from every provider in one place** (`polylogue`;
  `all-providers`). ChatGPT, 2025-05-20: "is there any project or anything
  really for scraping one's own AI chats from chatgpt.com, claude.ai, grok,
  google ai studio?" ChatGPT, 2025-05-02: "download their ochatgpt chatlogs
  (export), same with other ai chats". Spec: covered (`spec/sources/`).

### Agents, context and coordination

- **What enters an agent's context, controlled** (`polylogue`;
  `context-control`). ChatGPT, 2026-06-14: "to what extent could we really
  control context window('s content) thanks to hooks?" ChatGPT, 2026-07-05:
  "Notice what happens after the compaction. ... How far are we from being
  able to do better". ChatGPT, 2024-03-13: "LLMs would ideally have relevant
  content pasted into their ctx (but not too much". Spec: covered (briefings,
  `spec/surfaces/mcp-and-hooks.md`; compaction, `spec/analysis.md` §12).
- **One substrate for agent memory and the operator's own marks**
  (`polylogue`; `memory-substrate`). ChatGPT, 2026-06-14, answering "a
  typed, evidence-linked agent memory substrate": "Sure, tho also ofc we want
  to realize user's annotations and such on the same substrate ...
  highlights (incl color, maybe note) shoudl be KV". Spec: covered in the
  main (notes, inline markers and checked claims, `spec/user-data.md`);
  highlight colour is not.
- **Agent memory generated from a codebase** (`polylogue`; `code-memory`).
  claude.ai, 2026-05-25: "whether it would maybe be possible/useful to
  generate such agent memory programmaticallly, I mean specify what we want
  to index". Spec: partly (directory briefings); an index of code facts is
  not covered.
- **Polylogue as the substrate for coordinating agents** (`both`;
  `coordination-substrate`). ChatGPT, 2026-07-09: "how could fable better
  orchestrate across both projects, and both codex and claude? ...
  polyllogue seems potentially well positioned to host such coordination.
  ... to be sorta substrate." The same thread asks to "orchestrate
  efficiently across both OpenAI and Claude" around quota. Spec: live state
  and peers exist (`spec/model.md`); work threads, a shared blackboard and
  quota-aware routing are set aside undesigned (`SET-ASIDE.md`, Python
  surfaces).
- **A cockpit: what is happening now, in five seconds** (`both`;
  `cockpit`). claude.ai, 2026-05-01, the Polylogue Cockpit design: "Can the
  user tell what is happening RIGHT NOW within 5 seconds of looking at the
  cockpit?"; "Can they tell what is unhealthy / stale / waiting?", with a
  "stuck-promotion lane" and "kill/fork/handoff actions". Spec: partly (live
  state, `spec/model.md`; morning check, `spec/workflows.md` §2); acting on
  an agent from the view is not covered.
- **Issues that carry their own solution** (`both`; `pre-solved-work`).
  ChatGPT, 2026-06-13: "we want to make issues basically contain their own
  solutions, almost -- we want to pre-solve them, so that ideally coding
  agent has not much to do". ChatGPT, 2026-07-15: "produce an
  implementation-grade handoff". Spec: partly (`spec/workflows.md` §7 and
  takeover briefings); assembling a task packet from the sessions and notes
  behind it is not covered.
- **A personal secretary agent fed by the archives** (`both`;
  `secretary`; related to the Sinex section's `intentions-vs-actions`).
  ChatGPT, 2025-11-03, a design brief listing "chat archives
  (/realm/data/chatlog/ + Polylogue outputs)" as inputs, to "Seed Taskwarrior
  with the actionable backlog by harvesting tasks from existing docs" and
  "cross-check actual behaviour ... against tracked work". Spec: not covered
  (obligations raised in sessions are not extracted).
- **Better prompts for agent loops, from history** (`polylogue`;
  `prompt-improvement`). claude.ai, 2026-06-30: "coding agents need much
  better prompts for these loops". ChatGPT, 2026-06-28: "optimize such a
  prompt for rapid nimble autonomous project improvement loop". Spec:
  covered (`spec/workflows.md` §20).
- **A second opinion from another model** (`polylogue`; `second-opinion`).
  claude.ai, 2025-11-11: "I want to get a second opinion (of another LLM).
  I'll show it the project, as well as these discussio[ns]". Spec: partly
  (blind comparison is planned, `spec/user-data.md`, Judgment); packaging a
  session for another model is the export format.

### Projects and the work itself

- **A project's trajectory and velocity, from its history** (`both`;
  `trajectory`). ChatGPT, 2026-05-29: "analyze development trajectory, what
  progress has been made over" a month. ChatGPT, 2026-06-09: "so that
  trajectory / speed over time is most legible". Spec: covered
  (`spec/workflows.md` §17).
- **What the project would be if its plans were carried out** (`both`;
  `plans-projected`). ChatGPT, 2026-05-26: "what the project would be if
  extant plans were fully realized as they are right now". The same thread:
  "complete top-down view of the entire thing". Spec: not covered; this is
  the outline layer the Spec is adding (`polylogue-spec/outline/`, planned).
- **Lifetime spend, priced as the API would have charged** (`polylogue`;
  `lifetime-cost`). ChatGPT, 2026-06-23: "an estimate of API-equivalent
  costs, lifetime ... breakdown per model, read/write, cache/no-cache".
  Spec: covered (`spec/analysis.md` §1).
- **The archive as evidence of work for others** (`polylogue`;
  `proof-of-work`). ChatGPT, 2026-06-09: "I could also point at volume of my
  coding agent session logs, which is tens of gigabytes at this point."
  Spec: partly (shared excerpts, `spec/workflows.md` §19); a public summary
  of work with nothing private in it is not covered.
- **Speech turned into a living text** (`both`; `voice-text`). The
  operator's notes (`exocortex-metaanalysis.md`): "voice input system which
  would use LLM to maintain/develop semi-persistent text which represents
  what I'm talking about". Spec: not covered.

### Strongest uses not yet covered

1. `chats-vs-project`: reconcile chats with a project's current state
   (still relevant, done, unrealized).
2. `distill-thread`: a kept distillate per long session, tied to its source
   and re-made as it grows.
3. `coordination-substrate`: Polylogue as the coordination substrate for
   agents across projects, harnesses and quotas.
4. `idea-ledger` and `stance-map`: the operator's ideas and views across all
   chats, connected and checked for contradictions.
5. `secretary`: obligations harvested from sessions into a tracked backlog
   with provenance.
6. `plans-projected`: the system as its plans would make it (the outline
   layer).
7. `my-voice`, `prompted-review`, `voice-text`: generation from the
   operator's own corpus.

## Machine and life uses (sinex session)

Method: the operator's own unpasted turns in the web threads, by concept (attention, lifelogging, recall, sleep and health, money, places, media, browsing, journals, reports, patterns, nudges, habits: 1,382 messages, 2023-03 to 2026-07), then every turn stating an intent ("I want", "should be able to", "I'd like", "would be useful") together with a capture or tracking word (169). Most concept hits are questions about the operator's own life, not designs, and are left out. Each use below is in `sinex-spec/vision/uses.toml` under the id given.

- **Everything recorded, and the past reachable** (`sinex`; `day-recall`, `back-after-absence`). The founding statement: 2023-10-08, "basically track/monitor everything. I should be able to see what was happening in the past, be able to access it and so on. So, in other words, it should incorporate Quantified Self, Lifetracking (for digital stuff and beyond)".
- **The path through the web, not only the visits** (`sinex`; `browsing-trail`). 2023-04-11, an extension that records "(consequent locations, timestamps, and transition reasons)"; 2024-01-13, "I want to track user's movement through tabs, and within a tab (clicking on links leading to other URLs etc.)"; 2025-04-21, "I dont' want just webpages and how they link to each other, I want visits and how they link"; 2025-04-27, "as much web browsing-related data as possible ... For life-logging".
- **Screen and audio, continuously and cheaply** (`sinex`; `screen-audio-text`). 2023-07-16, "record exact content of the screen ... when there's not a lot going on on the screen, it shouldn't use a lot of resource"; 2024-02-17 and 2025-04-19, continuous capture of every audio input and output; 2025-06-04, "per-window recording ... OCR in realtime".
- **Reading as a timeline** (`sinex`; `reading-order`). 2024-01-16, "I'd like to somehow track what I'm reading, in what order. Basically have a timeline."
- **A permanent clipboard** (`sinex`; `clipboard-recall`). 2024-06-24, "keep a permanent log / history of what was selected, or held in the clipboard ... 'selected' part - PRIMARY".
- **Commitments fed by captured data** (`sinex`; `commitment-feed`). 2024-01-22, sleep tracked by the watch and wired to a commitment service (a Beeminder integration), with the computer acting on the schedule.
- **Files and media by tags, not folders** (`sinex`; `tagged-library`). 2025-04-26, after an article on tags over hierarchies: stop using a dedicated tagging app and "model metadata in exocortex".
- **The exocortex as an interface, not an application** (`both`; `ambient-interface`). 2025-04-20, "I want it to be an exocortex, meaning I'd use it basically as an interface ... It's constant part of the extremely general workflow."
- **Dashboards and summaries from the database** (`sinex`; `weekly-report`). 2025-05-02, "continuous aggregates, dashboards, summaries", stored rather than recomputed from raw files each time.
- **A period reconstructed hour by hour** (`sinex`; `reconstruct-period`). 2026-03-31, from activity-tracker data, "a narrative of what was my activity over time ... day by day, hour by hour really".
- **Projects and goals derived from the journal** (`both`; `project-ledger`). 2025-04-25, from the raw log, "Completed projects/tasks/goals, open ones, relationships between them, priority".
- **A system to live by** (`sinex`; `intentions-vs-actions`). 2025-09-01, "I want to implement a system to live by, making myself more coherent. I'm in the process of building Sinex the exocortex - it likely heavily overlaps."
- **Phone data logged** (`sinex`; `phone-watch`). 2025-11-02, rooting a phone, among the reasons "logging data".
- **Not tied to one machine's layout** (`sinex`; a constraint on every use). 2026-06-09, "sinex is not supposed to be specific to my own setup as it is right now, with its paths and whatnot."
