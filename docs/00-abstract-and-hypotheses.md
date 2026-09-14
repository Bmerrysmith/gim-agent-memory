# GIM — Abstract and Hypotheses

## Working abstract

LLM agents increasingly operate as short-lived processes: an agent is constructed for a task, accumulates hidden state and context, terminates, and is replaced. Weights are frozen at inference, so nothing the agent discovers during an episode survives except what is written to external storage. A growing body of work addresses this by distilling experience into retrievable memory, but these systems are evaluated on task success and token cost, and none report what happens when persistent storage is bounded. Storage is treated as free.

We study agent memory under an explicit hard byte budget. We introduce Geometric Intergenerational Memory (GIM), a write-and-forget policy layered on conventional dense retrieval, in which trajectories from both successful and unsuccessful agents are distilled into typed memory items carrying outcome polarity and provenance, embedded, consolidated into prototypes that retain sufficient statistics, and evicted by a compactor that enforces `bytes(M_n) <= B` at every generation boundary under a coverage constraint protecting rare failure families. We estimate the marginal utility of a memory by paired-seed counterfactual rollouts and learn a cheap predictor of that estimate from geometric and provenance features.

Our primary contribution is a measurement: a controlled comparison of memory policies at equal persistent bytes, in an environment with exact ground truth, sweeping budget levels across no-memory, raw-trajectory, reflection, success-only, FIFO, random, and utility-aware conditions. We report success rate, repeated-failure rate, failure-family coverage, and success-per-byte, with online management cost, offline labeling cost, and storage reported separately. Secondary contributions are a formal definition of the generation reset boundary with a structural isolation test, and evidence on whether negative memories and measured information gain predict downstream utility better than the source episode's reward.

(Draft. The claim to defend is the measurement, not the architecture. Revise once Gate 5 returns a result.)

---

## Primary hypothesis

H1 — Budget-aware retention beats budget-naive retention at equal bytes.
At a fixed persistent-byte budget B, a retention policy that scores memories by predicted downstream utility, information gain, and failure-family coverage yields higher task success per byte than FIFO, random, or recency-based eviction.

Falsified if: across budget levels, utility-aware retention is statistically indistinguishable from FIFO at equal bytes. This is the Gate 5 kill test.

## Secondary hypotheses

H2 — Failure memories carry transferable information.
Memories distilled from trajectories with terminal reward 0 improve successor performance relative to a success-only memory of equal byte cost, primarily by reducing repeated-failure rate rather than by shortening successful paths.

Falsified if: the negative-memory ablation is indistinguishable from success-only at equal bytes, or if negative memories improve repeated-failure rate while reducing overall success.

H3 — Provenance features predict utility better than source reward.
In a model predicting counterfactually estimated marginal memory utility, measured information gain and outcome polarity carry higher permutation importance than the source episode's terminal reward.

Falsified if: source reward dominates, which would mean success is the only signal worth storing and the polarity machinery is unjustified.

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

## The measurement that does not exist yet

    eta = (SR_policy - SR_nomemory) / MB_persistent

Every system in the related work reports the numerator. None report the denominator.
