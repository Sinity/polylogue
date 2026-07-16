# Local LLM and retrieval stack for a 40GB bilingual personal AI-session corpus

## Bottom line

For this box and this corpus, the stack I would actually deploy first is **Qwen3 8B Q4_K_M for generation**, **BGE-M3 for first-stage retrieval**, and **Qwen3-Reranker-0.6B for second-stage reranking**. That recommendation is not the absolute-best-benchmark stack; it is the best **quality / memory / multilingual / code-heavy / operational simplicity** stack for a single machine with an i7-13700K and about 10 GB of VRAM. Qwen3 8B gives the best balance of multilingual competence, coding usefulness, and local fit among the generator candidates; BGE-M3 is the most practical first-stage retriever because it gives you **dense + sparse + ColBERT-style signals in one multilingual model**; and Qwen3-Reranker-0.6B is the best small reranker in the compared official numbers, especially on multilingual, long-document, and code retrieval. citeturn26view0turn28view1turn34view2turn49view0

The most important operational judgment is this: **for “narrate a numeric measure without inventing facts,” the quality floor is not a reasoning model; it is a sufficiently competent instruct model used in a constrained extract-then-verbalize pipeline.** In practice, I would treat **Gemma 3 4B** as the lower acceptable local floor for that task if the facts are pre-extracted and handed to the model in a structured object, but I would use **Qwen3 8B** as the default because it gives more headroom on bilingual and code-adjacent summarization. DeepSeek-R1-Distill-Qwen-7B is stronger on reasoning benchmarks, but that is not the same thing as being the best narrator of already-computed measures; its reasoning orientation is more likely to add latency and unnecessary elaboration. That last point is inference, but it is grounded in the fact that DeepSeek’s 7B distill is explicitly a reasoning distill, while Qwen3 explicitly supports switching between thinking and non-thinking modes for task control. citeturn31view1turn27view2turn22academia1

One more strong conclusion: **do not treat Voyage-3 as the cloud gold standard anymore**. Voyage’s own docs mark the 3-series as an older generation and state that the newer listed models are better in quality, context length, latency, and throughput. If you keep a cloud path, the real current baselines are **voyage-4-lite / voyage-4** for general multilingual retrieval and **voyage-code-3** if code retrieval dominates enough to justify a specialized index. citeturn16view0turn18view2

## Generative model decision matrix

The cited facts below are evidence. The **tokens-per-second bands are operational estimates** for a 13700K plus a 10 GB-class NVIDIA card, because equivalent published same-hardware llama.cpp / Ollama benches are sparse and inconsistent across drivers, offload choices, and context sizes.

| Model | Practical local form | Evidence snapshot | Estimated practical decode | Verdict for measure narration |
|---|---|---|---|---|
| **Gemma 3 4B** | `gemma3:4b` in Ollama, Q4_K_M, **3.3 GB** | Ollama ships a 4.3B-parameter Q4_K_M artifact at 3.3 GB; Gemma 3 supports over 140 languages and a 128K context window for the 4B model; Google’s Gemma card reports the 4B pretrained model at **MMLU 59.6** and **HumanEval 36.0**. Gemma 3 QAT variants are described as preserving similar BF16 quality at roughly **3x less memory**. License is **Gemma Terms of Use**, not Apache. citeturn26view1turn27view0 | **30–55 tok/s** | **Acceptable floor** if you do strict grounding and structured verbalization. Fastest comfortable local option. |
| **Qwen3 8B** | `qwen3:8b` in Ollama, Q4_K_M, **5.2 GB** | Ollama ships Qwen3 8B at 5.2 GB Q4_K_M under **Apache 2.0**. Qwen says the dense 8B model supports **119 languages and dialects**, and Qwen3 explicitly supports both **thinking** and **non-thinking** modes for task control. Small external studies also found Qwen3-8B notably strong on low-resource multilingual adaptation and factual correctness in RAG-style evaluation. citeturn26view0turn28view1turn29academia3turn30academia1 | **18–35 tok/s** | **Best default**. Best balance of multilingual strength, code usefulness, and controllable verbosity. |
| **DeepSeek-R1-Distill-Qwen-7B** | `deepseek-r1:7b` in Ollama, Q4_K_M, **4.7 GB** | Ollama’s 7B artifact is a **7.62B** Q4_K_M model at 4.7 GB under **MIT**. The model card reports **AIME 2024 pass@1 55.5**, **MATH-500 pass@1 92.8**, **GPQA Diamond 49.1**, and **LiveCodeBench pass@1 37.6** for the 7B distill. Independent work on quantized reasoning models also found that lower-bit quantization can introduce accuracy risks, especially on reasoning tasks. citeturn27view1turn31view1turn22academia1 | **Raw decode similar to other 7–8B models, but realized answer throughput is often lower because outputs are longer** | **Not my default**. Strong reasoner, but the wrong default temperament for concise, non-inventive measure narration. |
| **Gemini 3.1 Flash-Lite** | Cloud baseline | Google’s current stable “Flash-Lite” tier is **Gemini 3.1 Flash-Lite**, described as the most cost-efficient model for high-volume agentic tasks and simple processing. Standard paid pricing is **$0.25 / 1M input tokens** and **$1.50 / 1M output tokens**. citeturn45view0turn44view0 | N/A | **Best cheap cloud baseline**, but not local. Useful only as a reference point and for occasional overflow. |

The defended recommendation here is straightforward: **use Qwen3 8B, and explicitly run it in non-thinking mode for classify / summarize / narrate-measures jobs**. Qwen3’s multi-mode design is unusually relevant to your use case because your hard problem is not olympiad reasoning; it is **staying grounded while converting measurements and retrieved evidence into compact prose**. For that job, “fewer thoughts” is often better than “more reasoning traces.” citeturn27view2

For the **quality floor**, I would draw the line this way. If the model is only verbalizing a precomputed JSON object such as:

```json
{
  "metric": "session_count",
  "value": 482,
  "period": "2025-Q4",
  "delta_vs_prev": -0.07,
  "evidence_ids": ["chunk_1022","chunk_9833"]
}
```

then **Gemma 3 4B is viable**. If the model must also reconcile bilingual prose, code snippets, and partially structured context before verbalizing the result, **Qwen3 8B is the safer local minimum**. That is an inference, but it fits the evidence: Gemma 3 4B is impressively capable for size, while Qwen3’s multilingual positioning and local-use ergonomics are stronger for mixed Polish-English archives. citeturn27view0turn28view1

## Embedding and reranker decision matrix

The retrieval side matters more than the generator here. Your corpus is bilingual, code-heavy, and likely full of identifiers, session names, branch names, prompt fragments, and concept drift across revisions. That means you need a retriever that handles **semantic matching** and **literal matching** well. This matters even more in Polish, where BM25-style lexical baselines degrade more than they do in English because of morphology; BEIR-PL explicitly reports that BM25 scores are significantly lower in Polish than in English. citeturn47academia1

| Candidate | Evidence snapshot | Practical memory/runtime profile | Recommendation |
|---|---|---|---|
| **nomic-embed-text-v1.5** | English long-context embedder, **0.1B params**, **8192** tokens, Apache-2.0. Official card reports **MTEB 62.28** at 768 dims, **61.04** at 256 dims, with Matryoshka-style dimension reduction. Its technical report positions it as an **English** long-context model that beats Ada-002 and text-embedding-3-small on short-context MTEB and LoCo. citeturn12view3turn40academia1turn48view3 | Roughly **0.4 GB** FP32 weights; very easy to run; storage-efficient because of dimension truncation. | **Not first choice** for this corpus. Excellent efficiency, but the public evidence you can actually cite is too English-centric for a Polish+English archive. |
| **BGE-M3** | MIT-licensed, **1024 dims**, **8192** tokens, **100+ languages**, and uniquely supports **dense, sparse, and multi-vector retrieval in one model**. The BGE paper claims new SOTA on multilingual and cross-lingual retrieval, and the model card explicitly recommends **hybrid retrieval + reranking**. In a recent Khmer-domain RAG study, **BGE-M3 outperformed Qwen3-Embedding and Jina-v3** on the retrieval phase. citeturn13academia0turn34view2turn48view0turn30academia1 | About **1.1–1.3 GB** in FP16-class memory by parameter count inference; still comfortable on CPU or GPU; one-pass generation of dense and sparse signals is operationally attractive. | **Best production first-stage local retriever** for this machine. |
| **gte-multilingual-base** | **305M params**, **768 dims**, **8192** tokens, **75 languages**, Apache-2.0. Alibaba positions it as an encoder-only multilingual retriever that offers a **10x inference-speed increase** over previous decoder-based GTE models. The accompanying paper says its text encoder and reranker **match large BGE-M3 models** and do better on long-context retrieval benchmarks. The card also explicitly mentions MTEB results on **English, Chinese, French, and Polish**. citeturn33view1turn39academia0turn48view1 | About **0.6 GB** FP16-class memory by parameter count inference; very attractive if you prioritize speed and simplicity. | **Best simple alternative** if you do not want BGE-M3’s hybrid machinery. |
| **Qwen3-Embedding-0.6B** | **0.6B params**, **32K** context, **100+ languages**, Apache-2.0. Official card reports **MMTEB mean(task) 64.33**, **MTEB English mean(task) 70.70**, and explicitly positions the family for **multilingual, cross-lingual, and code retrieval**. citeturn36view0turn37view0turn37view1turn48view2 | About **1.2 GB** BF16-class memory. More expensive than GTE, still manageable. | **Strong light dense-only option**, especially if you want a single Qwen family on the retrieval side. |
| **Qwen3-Embedding-4B** | **4B params**, **32K** context, **100+ languages**, Apache-2.0. Official card reports **MMTEB mean(task) 69.45** and **MTEB English mean(task) 74.60**, with strong retrieval scores and explicit support for custom output dimensions. citeturn37view3turn38view0 | About **8 GB** BF16-class weights before runtime overhead. It can fit on a 10 GB card, but only awkwardly, and not as a comfortable always-on companion to your generator. | **Best local dense-only upgrade path**, but too heavy for my first recommendation on this box. |
| **Voyage-3 baseline** | Voyage docs list `voyage-3` as an **older model**, **32K** context, **1024 dims**, **$0.06 / 1M tokens** for the old pricing table, and explicitly say the newer listed models are **strictly better** than legacy ones in quality, context length, latency, and throughput. citeturn16view0turn17view0 | Zero local memory, but ongoing API cost and off-box data flow. | **Useful historical baseline, not a current target state**. |

For reranking, the official Qwen3 numbers are the cleanest small-model comparison in the source set. On Qwen’s published reranking evaluation, **Qwen3-Reranker-0.6B** scores **65.80 on MTEB-R**, **66.36 on MMTEB-R**, **67.28 on MLDR**, and **73.42 on MTEB-Code**. In the same table, **gte-multilingual-reranker-base** scores **59.51 / 59.44 / 66.33 / 54.18**, and **BGE-reranker-v2-m3** scores **57.03 / 58.36 / 59.51 / 41.38**. Qwen’s note matters: these reranker results are computed over the top-100 candidates from Qwen3-Embedding-0.6B, so they are not completely architecture-neutral, but the margin is large enough to matter. citeturn49view0

That is why my recommendation is **BGE-M3 retrieval, Qwen3-Reranker-0.6B reranking**. BGE-M3 gives you the most practical first-stage retrieval surface for a messy personal archive; Qwen’s reranker then fixes ranking precision where it counts.

## Recommended stack and dogfood plan

The concrete stack I would deploy first is:

- **Generator:** `qwen3:8b` in Ollama, Q4_K_M, defaulted to non-thinking mode for corpus-scale classify / summarize / narrate-measures work. citeturn26view0turn27view2
- **Retriever:** `BAAI/bge-m3` for first-stage retrieval, using **both dense and sparse outputs**. citeturn34view2
- **Reranker:** `Qwen/Qwen3-Reranker-0.6B` for top-50 or top-100 second-stage rescoring. citeturn49view0
- **Cloud comparison path:** keep one benchmark lane against **Voyage-3** only long enough to decide whether local wins on your own archive; if you stay cloud afterwards, migrate to **Voyage-4-lite / 4** rather than keep a legacy baseline. citeturn16view0turn17view0

The dogfood plan should be narrow, harsh, and cheap.

Start by indexing only a **hard slice** of the archive rather than the whole 40 GB. Pick a few thousand sessions that are known to contain branch / resume / fork relationships, plus a representative subset of code-heavy conversations and Polish-English mixed sessions. Build four retrieval lanes over the exact same chunks:

1. **FTS / BM25 baseline**  
2. **BGE-M3 dense-only**  
3. **BGE-M3 dense+sparse hybrid**  
4. **BGE-M3 dense+sparse hybrid + Qwen3-Reranker-0.6B**

That directly addresses the operator principle “prove vector beats FTS before building on it.” It also respects the BEIR lesson that **BM25 remains a robust baseline**, and BGE’s own card explicitly notes that BM25 is still competitive, especially on long-document retrieval. citeturn47academia0turn35view0

After that, add the generation lane only for tasks where retrieval already wins. The generator should **not** be used to invent measures, infer counts from prose, or reconcile ambiguous retrieval hits in one pass. Instead:

1. retrieval returns candidate chunks;  
2. a deterministic reducer computes the metric or assembles the evidence bundle;  
3. Qwen3 8B verbalizes only the computed object and cited evidence.

That design choice is inference, but it is the right one for your stated “narrate a numeric measure without inventing facts” requirement.

If that first local retrieval stack loses to FTS on your lineage-pair evaluation, I would **not** abandon local immediately. I would first swap the embedder to **Qwen3-Embedding-4B**, keep the same reranker, and rerun the same benchmark. The public benchmark evidence says Qwen3-Embedding-4B is materially stronger than BGE-M3 on multilingual MTEB aggregates, but it is much heavier operationally; that is why it is my **second** move, not the first. citeturn38view0

## Self-labeled archive evaluation

The right local proof here is a **self-labeled retrieval benchmark built from the archive’s own fork / resume lineage**, not a synthetic QA set.

The construction is simple.

Take any session that is clearly a descendant, continuation, branch, or rewrite of another session. Treat the newer session’s operator-facing summary, title, first prompt, or manually curated “query object” as the **query**, and treat the ancestor session or its key chunks as **positive documents**. The best positives are the ones that are **semantically near but lexically far**: the same design resumed under a different naming scheme, a Polish restatement of an English design, a code refactor that preserved intent but not identifiers, or a narrative rewrite of a numeric analysis. Those are exactly the cases where dense or hybrid retrieval should beat pure FTS.

For every query, record at least three positive sets:

- the exact parent session  
- sibling sessions in the same fork family  
- chunks manually marked as “same thread / same object / same operator concern”

Then construct **hard negatives** from near-time neighbors, same-language same-day sessions, or code sessions sharing generic libraries but not the same underlying object. This matters because easy negatives will flatter every retriever.

The metrics I would track are:

- **Recall@k** for k in {5, 10, 20, 50}  
- **MRR@10**  
- **nDCG@10** if you have graded relevance inside a lineage  
- **family hit rate**, meaning whether any member of the correct lineage family appears in top-k  
- **cross-lingual hit rate**, where the query language differs from the positive document language

That last metric is critical for your corpus.

This methodology is justified by the general IR literature in two ways. First, BEIR emphasizes evaluating retrieval systems across heterogeneous tasks and architectures rather than trusting a single benchmark regime. Second, recent work such as HAKARI-Bench shows that **small, repeated, same-condition retrieval benchmarks** are a practical way to compare models, quantization, dimensionality, and reranking settings during development. citeturn47academia0turn47academia2

A clean protocol would look like this:

1. Freeze a chunking recipe.  
2. Freeze a set of lineage-family queries.  
3. Run FTS, dense, hybrid, and reranked variants over the exact same corpus slice.  
4. Inspect failures manually, especially:
   - Polish inflection mismatches  
   - code-renaming events  
   - long-session continuation hops  
   - concept drift across rewritten plans  
5. Promote the winning stack only if it beats FTS by a **material** margin on your own positives, not just on public tables.

The main operational criterion I would use is this:

- **Do not adopt dense retrieval as the source of truth unless hybrid or dense beats FTS by at least ~10–15 percentage points on Recall@10 over lineage-family queries, or wins clearly on the failure classes you actually care about.**

That threshold is inference, but it is the right kind of inference: big enough to survive measurement noise and subjective cherry-picking.

## When local embeddings beat Voyage

On this corpus, local embeddings beat the current Voyage-3 path in three cases: **rebuild cost**, **privacy posture**, and **rebuildability of the embeddings.db tier**.

The cost case is the easiest to quantify. Voyage’s current docs price legacy **`voyage-3` at $0.06 per million tokens**. citeturn17view0 If your 40 GB archive corresponds to roughly **5B to 10B tokens** after text extraction and before overlap inflation — which is only a rule-of-thumb inference, not a measured fact — then one full re-embedding pass costs roughly:

- **$300 at 5B tokens** citeturn50calculator0  
- **$480 at 8B tokens** citeturn50calculator1  
- **$600 at 10B tokens** citeturn50calculator2

And that is before overlap, re-chunking, or repeated rebuilds. By contrast, the local marginal cost of running BGE-M3 or GTE on your existing machine is dominated by time and electricity, not API spend. That is why the financial break-even point arrives very quickly for a large personal archive.

If you stayed cloud but moved to the cheaper current tier, **Voyage-4-lite** is **$0.02 per million tokens** with a 200M-token free tier, which drops those same illustrative one-pass costs to roughly **$100 / $160 / $200** at 5B / 8B / 10B tokens. That is much better than legacy Voyage-3, which is another reason I would not keep Voyage-3 as the serious comparison target. citeturn17view0turn50calculator3turn50calculator4turn50calculator5

The privacy case is simpler. A personal AI-session corpus is likely full of prompts, logs, partial code, personal notes, internal naming schemes, and failure analyses. If you want the archive to remain **fully on-device and fully rebuildable from raw chunks plus model/version metadata**, local wins immediately. That is not a benchmark claim; it is an operational property.

The rebuildability case is the one I think matters most to your design. A good local embeddings tier is not just “vectors on disk.” It is a **rebuildable embeddings.db** with:

- chunk hash  
- source session id  
- index recipe version  
- embedding model id  
- output dimension  
- quantization / dtype  
- vector payload  
- optional sparse weights

That means you can delete the whole index and regenerate it deterministically from local source plus recipe. That is much harder to treat as a first-class invariant if the default path is an external API.

My cutoff would be:

- **Choose local now** if you expect full or near-full rebuilds, privacy matters, or the archive will evolve enough that you want deterministic regeneration.
- **Keep cloud temporarily** only if you need immediate convenience, your delta updates are small, and you are willing to treat retrieval as a service dependency rather than a local substrate.
- **If you keep cloud, do not keep Voyage-3**. Upgrade the baseline to a current Voyage model because Voyage itself now positions the 4-series as better than legacy 3-series across the board. citeturn16view0turn17view0

## Open questions for the operator

- How often will you **rebuild the whole index** versus append small deltas?
- Is the archive primarily queried by **natural-language prompts**, by **code / identifier lookups**, or by **lineage / continuation discovery**?
- Are your “narrate-measures” jobs mostly **single-metric descriptions** or **cross-session comparative narratives**?
- Do you want the retrieval stack to stay **single-index**, or are you willing to split into **general multilingual** and **code-specialized** indexes later if that wins locally?
- Is the 10 GB GPU expected to serve **generator and retriever concurrently**, or can those jobs be scheduled separately?

## What’s missing

- I could not verify a public, current **“gemma-embedding”** checkpoint matching the prompt wording, so I did not score it in the recommendation.
- Publicly accessible **same-hardware throughput benchmarks** for your exact 13700K + 10 GB VRAM configuration are sparse; the decode rates in this report are therefore operator-side estimates rather than cited benchmark facts.
- The publicly parsed BGE-M3 and GTE pages do not expose all the underlying table values in text form, so some of their comparative positioning relies on official claims and a smaller number of clearly surfaced benchmarks rather than a fully uniform score table.
- The real answer to “vector beats FTS” is still archive-specific. BEIR and BEIR-PL justify the evaluation philosophy, but your fork/resume lineage benchmark is the deciding test. citeturn47academia0turn47academia1