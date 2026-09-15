# Approved Phase A environment, option-2 evidence, and remaining proposals

The user approved the workshop environment and frozen policy for G0–G1, then selected option 2 for H3 on 2026-09-14: preserve episode-based outcome labels and use local evidence separately. Documents 10 and 11 record these implemented decisions. The information-gain estimator, utility estimand and Split design below remain proposals. The architecture's G0–G8 order and H1/H3 focus remain in force; these decisions do not establish inheritance or hypothesis results.

The provenance decision is settled: preserve provenance while its memory exists. The user prioritizes the original centroid-vector storage cost and minimizing storage, deployed token use, and latency while retaining useful transfer. The earlier four-group proposal is withdrawn. The option-2 decision below supersedes the earlier suggestion to redefine polarity as an individual tool result.

## Approved environment: a workshop with hidden tool rules

An agent must open three locks within five tool attempts. There are six lock types and four tools. Within one workshop, each lock type has exactly one working tool. The rule is stable across generations in that workshop, but different workshops have different hidden rules. Earlier agents can leave observations that help later agents avoid known failures or reuse known solutions.

This is deliberately small enough to replay and inspect by hand. It does not model general LLM competence. It supplies a controlled transfer problem with exact ground truth for the memory measurements in Phase A. Whether it actually supports the planned comparisons must still be tested at G5.

### The task definition

| Element | Proposed version 1 definition |
| --- | --- |
| Family | One workshop and its hidden six-entry lock-to-tool mapping. The family ID is an opaque label, not an encoding of the mapping. |
| Hidden state | The mapping, current task position and terminal state. Mapping values are sampled independently and uniformly from the four tools, subject to assigning distinct full mappings to different families in the study. |
| Task | An ordered sequence of three distinct lock types drawn from that workshop. Only the current lock is visible; the next lock is revealed after the current one opens. |
| Action | Apply one of four named tools to the current lock. There is no privileged inspect or ground-truth action. |
| Observation | Family ID, current lock type, tool names, attempts remaining, and the result of the immediately preceding action. A tool either opens the lock or produces a visible `wrong_tool` result. |
| Transition | Each valid tool attempt consumes one step. A correct tool advances to the next lock; an incorrect one leaves the lock closed. No hidden random transition is used. |
| Terminal reward | 1 if all three locks open within five attempts, otherwise 0. Intermediate outcomes are observations, not extra terminal reward. |
| Failure family | The visible rule failure `(family_id, lock_type, attempted_tool_id)` when the result is `wrong_tool`. Repeating that triple is a repeated failure. Exhausting the task horizon is recorded separately. |
| Generation | One fresh agent attempts one task, then terminates. Memory writes occur only afterward. |

Use versioned deterministic generation for both workshop maps and task sequences. Save the generator version and seeds. Generate the full episode schedule before evaluating policies, and give all conditions exactly that schedule. Task selection must not adapt to which retention policy is winning.

For an initial review example, a hidden workshop may have `triangle → blue tool`, `circle → red tool`, and `square → green tool`, among its six rules. A failed first episode might discover that the blue tool opens the triangle lock but run out of attempts later. The successor can use that valid positive observation even though its source episode received reward 0.

### A simple frozen agent policy

Maintain a four-element set of possible tools for the current lock. Start from all four each time a new agent encounters a lock. Remove tools ruled out by current observations and compatible retrieved `works=false` assertions. A compatible `works=true` assertion can identify the working tool. Either assertion can come from either source-outcome quota. Query memory using the family ID and current lock type, with the architecture's separate source-success and source-failure quotas.

Choose uniformly among the remaining possible tools using a deterministic random draw. Sort the remaining tools by stable tool ID before mapping the draw to that ordered list; Python set iteration order must not determine the selected tool. The implementation uses a counter-based SHA-256 draw keyed by policy seed and decision number, with rejection to avoid modulo bias. Paired runs can use the same random inputs even when their available actions differ; this does not make the two policies take identical actions.

If memory evidence conflicts, ignore the conflicting inherited assertions for that lock, log the conflict, and fall back to the within-episode observations. The proposed environment has no changing rules, so generated evidence should not conflict. A conflict is still useful for testing the memory interface and consolidation rules. An empty candidate set likewise falls back to all tools not disproven in the current episode.

The agent cannot read the hidden mapping, evaluator annotations, another agent's scratchpad, training labels, or experiment logs. Each fresh agent has an empty transient belief state. Its only cross-generation input is retrieved memory from the budgeted store.

### What transfers, and what is held out

Transfer occurs between episodes of the same workshop: a tool rule learned for one task can help a different three-lock task later. Workshop IDs remain in memory scope so evidence from one mapping is not applied to another. Rotate through multiple active workshops in a schedule fixed before the experiment; use a declared nonuniform schedule when testing whether retention preserves rarer failure families. This proposal does not yet set the rarity threshold or protected-family quota.

For predictor evaluation, separate entire workshops into training, validation, and test groups before generating trajectories or labels. All origin episodes, memories, utility rollouts and descendants of a workshop stay in that group. No family identity enters the predictor as a feature. Validation chooses model settings and any thresholds; the test group is used for final evaluation. Keep the split manifest and all seeds in the run record.

A review-sized fixture can use a few workshops and episodes, but its scores are only a demonstration. Set the final family count, sample size, mixture and stopping rule before collecting the H1/H3 evaluation data. Do not call a small fixture a powered study.

## Approved evidence and polarity; proposed information gain

The workshop distiller produces one typed assertion per observed tool attempt: either `tool works for lock` or `tool does not work for lock`, with workshop scope and supporting event ID. It checks the visible trajectory's internal consistency and does not consult hidden ground truth. It retains repeated evidence with its own citation; no unapproved novelty filter or omission rule is applied. Information gain is currently `None`, not a fabricated zero.

Under approved **option 2**, `outcome_class` remains the entire source episode's success/failure. The existing `k+` quota retrieves source-success memories and `k-` retrieves source-failure memories. Hard source-outcome merge compatibility remains part of the architecture. The source episode's actual terminal reward remains in provenance. No new `source_outcome_class` field duplicates the existing label.

The separate local evidence sign is the existing content field `works`. A failed episode can yield `works=true`, and a successful episode can yield `works=false`. H3 now compares local evidence and measured IG with source reward; it does not claim binary source outcome contributes information independent of reward. All experimental arms keep the same local facts in content. The comparison tests using those facts in prediction, not adding new facts or changing retrieval quotas. H2's success-only baseline still selects by source terminal reward.

Each observed `wrong_tool` carries the failure-family triple `(family_id, lock_type, tool)` encoded as a compact JSON string in the existing `failure_family` field. This is independent of the source episode's success/failure label, so a successful episode's warning can still satisfy a declared coverage requirement. Every tag and citation counts toward B. Which families qualify as rare and their scientific quotas remain unspecified.

### The finite posterior and information gain

For one lock, the evaluator's initial hypothesis set contains four possibilities: each tool might be the unique working one. The prior is uniform. For all six locks, the workshop has `4^6 = 4,096` possible mappings before observations. Because the prior factorizes across locks, the evaluator can update each four-element set independently; exhaustive enumeration is unnecessary.

A wrong-tool observation removes that tool from the lock's possible set. A correct-tool observation leaves only that tool. The proposed observed information gain is the entropy reduction in bits:

`IG(event) = log2(number of possibilities before) - log2(number after)`

For example, an initial failure reduces four possibilities to three and yields `log2(4/3)` bits. A success on the next attempt reduces three possibilities to one and yields `log2(3)` bits. Observing an already-known fact contributes zero bits. These are exact values for this declared deterministic likelihood and uniform posterior, not a proxy based on embedding novelty or reward.

Define this posterior from the evidence actually available to the source agent: compatible retrieved assertions plus observations already received in that episode. The evaluator replays that evidence; it does not initialize the posterior from the actual hidden answer. Contradictions use the same fallback rule as the agent. Compute an event's IG when its observation arrives, and freeze it in provenance before any future utility rollout. One candidate cites one observed event, so no cross-event aggregation choice is needed initially. Repeated evidence would receive its measured IG, including zero, once this estimator is implemented.

Keep the full hidden map inside the evaluator for correctness checks only. The agent sees its tool feedback; it does not receive counterfactual labels, unseen tool results, or future task sequences. In later phases, exact IG may be unavailable. Approval of this Phase A calculation does not authorize claiming exact IG for an LLM environment.

### Paired utility for H1 and H3

At G6, sample future tasks from the candidate's workshop using an independently generated, predeclared utility schedule. Freeze a background-memory snapshot M and candidate m. For every utility seed, construct two fresh agents and run the same downstream task and action horizon with M and with M plus m. Their random draws use the same keys. Save the paired terminal-return differences, mean and standard error. Do not distill or retain new memories between the pair's two arms.

For the first utility-label study, the proposed estimand is the candidate's **direct addition effect** without eviction. Use the same retrieval quotas and frozen agent policy, and explicitly record when the added-memory arm exceeds the online cap. These rollouts are offline measurement, not successful online generation boundaries. H1's actual policy experiment still enforces B at every boundary. A subsequent compaction-aware study would answer a different question and must label its estimand separately. This choice is part of the proposal for review.

The predictor can use frozen provenance and geometry available at write time, including observed IG, local evidence sign (`works`), source terminal reward, novelty, duplicate support and declared coverage features. It must not receive the hidden mapping, later returns, test-family IDs or labels derived from future success. Use held-out families for permutation importance and report uncertainty. IG, local evidence and reward can still be correlated; do not infer causality or claim H3 merely because one particular model ranks them favorably.

The environment does not choose the research embedder, H1 score weights, statistical thresholds or budget treatment. Those remain explicit decisions in document 05. Its scope is to make the task, transfer mechanism and evidence semantics reviewable.

## Centroid storage replaces the four-group proposal

Keep one float32 centroid per prototype: `4d` vector bytes, or **1,536 bytes
at 384 dimensions**, with compact scalar statistics. The former four-group
proposal and its approval request are superseded by the user's storage priority.
Do not add stored representative vectors or vector-sum arrays.

[Centroid storage design](08-centroid-storage-design.md) explains normalization,
rounding, scalar statistics, provenance and the remaining Split choices. A single
centroid plus scalars cannot recover discarded member assignments. The document
separates that limit from alternatives that supply the missing evidence explicitly
and charge it to B. No alternative is silently substituted for `Split(target)`.

## Remaining review

The workshop environment/policy and option-2 source/local evidence separation are
approved and implemented. The research embedder, information-gain calculation and
utility estimand remain separate decisions. The basic terminal distiller leaves
IG unavailable; it does not complete the scientific G2–G6 program.

The accepted storage target is one float32 centroid per prototype. A concrete
Split algorithm still needs agreement and validation at G4 before G5/G6 results
can be claimed. The architecture remains unchanged; H3 and its matching abstract
sentence have the explicit option-2 amendment documented in file 11.
