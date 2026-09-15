# GIM — Abstract and Hypotheses

## Central research question

> Under an audited persistent-byte cap, does utility-aware retention improve successor-agent performance, and do local evidence and information gain predict that utility better than source-episode reward?

Adopted by the user on 2026-09-14. H1 addresses retention and successor performance; H3 addresses prediction of memory utility. This adoption establishes the research question. The encoder, information-gain estimator, utility estimand, Split design and final statistical protocol remain separate decisions.

## Revised working hypothesis and status

Under an audited persistent-byte cap, utility-aware retention improves successor-agent
performance, and local evidence and information gain predict marginal memory utility
better than source-episode reward. This is the claim to test, not an established result.

H1 and H3 below retain identifiers for the two parts of this revised question.
The other hypotheses are supporting or exploratory directions; they do not replace
the central claim or require failed episodes to be inherently more valuable.
The workshop engineering benchmarks exercise the memory loop and byte accounting.
They have not evaluated utility-aware retention or fitted the proposed predictors,
so they neither establish nor refute the revised hypothesis.

## Working abstract

LLM agents can operate as short-lived processes whose inference weights remain frozen. Experience survives the generation boundary through external memory. Prior work already studies byte-budgeted memory write policies and value-per-byte retention, including [WritePolicyBench](https://arxiv.org/html/2602.02574v1) and [Forget to Improve / CURATOR](https://arxiv.org/html/2606.25115v1). GIM investigates whether retained experience improves fresh successor agents under an audited persistent-byte cap and which source evidence predicts that benefit.

Geometric Intergenerational Memory (GIM) is a proposed write-and-forget policy layered on conventional dense retrieval. Trajectories from successful and unsuccessful agents are distilled into typed memory items carrying source outcome and provenance. The architecture embeds and consolidates compatible items into prototypes, then enforces `bytes(M_n) <= B` at every generation boundary under a declared failure-family coverage constraint. Content, vectors, retained provenance, metadata and index overhead enter the byte audit. The planned utility pipeline estimates marginal memory utility with paired-seed counterfactual rollouts and fits a cheap predictor from features available before those downstream outcomes.

The primary planned measurement compares utility-aware retention with FIFO, random and recency-based retention at the same declared persistent-byte cap in an environment with exact ground truth. Actual occupied bytes are reported separately from the cap. The study will report successor task success, repeated failures, failure-family coverage and success-per-byte, separating online management cost, offline labeling cost and storage. H3 tests whether local evidence and measured information gain predict counterfactual utility better than the source episode's reward, while preserving episode-based retrieval quotas and source-outcome merge compatibility. The generation reset boundary and its isolation checks are part of the experimental contract.

(Working abstract aligned with the adopted question. G5 tests inheritance; G6 supplies the utility measurements for H1 and H3. No inheritance or H1/H3 result, or claim of being the first byte-budgeted memory study, is established.)

---

## Primary hypothesis

H1 — Budget-aware retention beats budget-naive retention at equal bytes.
At a fixed persistent-byte budget B, a retention policy that scores memories by predicted downstream utility, information gain, and failure-family coverage yields higher task success per byte than FIFO, random, or recency-based eviction.

Falsified if: across budget levels, utility-aware retention is statistically indistinguishable from FIFO at equal bytes. H1 is evaluated at G6 after the G5 inheritance gate.

## Secondary hypotheses

H2 — Failure memories carry transferable information.
Memories distilled from trajectories with terminal reward 0 improve successor performance relative to a success-only memory of equal byte cost, primarily by reducing repeated-failure rate rather than by shortening successful paths.

Falsified if: the negative-memory ablation is indistinguishable from success-only at equal bytes, or if negative memories improve repeated-failure rate while reducing overall success.

H3 — Local evidence and information gain predict utility better than source reward.
In a model predicting counterfactually estimated marginal memory utility, measured information gain and local evidence sign carry higher permutation importance than the source episode's terminal reward. In the workshop, local evidence sign is the existing `works` fact: whether the cited tool attempt opened its scoped lock.

Episode-based outcome polarity remains the source episode's success/failure label and continues to govern retrieval quotas and source-outcome merge compatibility. It is not redefined as local evidence sign. In this binary-reward environment, source outcome and source reward contain the same information; they must not be presented as independent predictors. All comparison conditions retain the same local facts in memory content; the comparison tests using those facts as predictor features.

Falsified if: source reward dominates the local-evidence and information-gain features under the declared held-out evaluation. This result would concern the tested predictors; it would not establish that failed episodes contain no useful information or replace the separate H2 test.

(Option 2 amendment approved by the user on 2026-09-14. See [decision and implementation](11-source-outcome-and-local-evidence.md).)

H4 — Consolidation dominates quantization under a byte budget.
At equal bytes, reducing the number of items by merging into prototypes preserves more task performance than reducing bytes per item by lowering vector precision.

Falsified if: quantization alone matches or beats consolidation, which would make the geometric machinery unnecessary and the contribution purely a compression result.

H5 — Representation cost is the dominant storage term, and symbolic typing is competitive.
A dense embedding is roughly an order of magnitude more bytes than the distilled text it indexes. At equal bytes, typed relational memory (compatibility, contradiction, precondition, supersession) is competitive with embedding-based retrieval for agent inheritance, with a measurable crossover in budget.

Falsified if: typed memory underperforms at every budget level tested.

H0 — The null worth stating.
No memory condition beats no-memory at any tested budget. If this holds in Phase A, the environment does not support transfer and the generator must be redesigned before anything else is interpretable.

---

## The central formal claim

An agent generation is a construction-to-termination interval across which transient state does not persist:

    R_reset(h_i, KV_i, scratchpad_i) -> empty,   while M_{i+1} persists

This is enforced structurally rather than asserted: the agent object is constructed fresh per generation, holds no reference to any predecessor, and has no write path to the store. The isolation test is a first-class experiment, not a unit test detail.

## Planned efficiency measurement

    eta = (SR_policy - SR_nomemory) / MB_persistent

This is a planned GIM measurement, not a claim that earlier work omits storage costs. Declare how actual occupied bytes across generations enter `MB_persistent`, and report them alongside the configured cap. The no-memory condition supplies the reference success rate; its own zero-byte efficiency ratio is undefined. The final comparison and statistical decision rules remain to be specified in the research protocol.
