# Implementation, validation, and research decisions

The central research question, adopted by the user on 2026-09-14, is:

> Under an audited persistent-byte cap, does utility-aware retention improve successor-agent performance, and do local evidence and information gain predict that utility better than source-episode reward?

The implementation follows [03-architecture-diagram.md](03-architecture-diagram.md). H1 and H3 in [00-abstract-and-hypotheses.md](00-abstract-and-hypotheses.md) address the question's retention and utility-prediction parts. The earlier approved option 2 preserves source-episode outcome polarity while using local evidence separately for H3. Adopting this question does not select an encoder, information-gain estimator, utility estimand, Split algorithm or statistical decision rule.

This direction resolves the build-order ambiguity: G5 checks whether memory supports inheritance at all; G6 adds counterfactual utility estimation and the predictor needed to evaluate H1 and H3. The hypotheses document's gate reference is aligned with this implementation order. A successful infrastructure test is not evidence that a research hypothesis holds.

The user also resolved provenance retention: preserve provenance while its memory exists. Retained memories keep immutable provenance; evicting a memory removes its provenance from the agent-visible store as part of the same operation. Any separate historical audit records remain inaccessible to agents. The user subsequently prioritized one float32 centroid's storage cost, useful transfer, lower deployed token cost, and speed. The four-group prototype proposal is withdrawn. See [Phase A proposal](06-proposed-phase-a-and-split.md) and [centroid storage design](08-centroid-storage-design.md) for the remaining research choices; neither code presence nor a proposed design constitutes a passed gate.

## What the components mean and how they connect

An **episode** is one attempt at a task. A **generation** is the lifetime of its agent. The agent receives observations and retrieved memories, chooses actions, and produces a trajectory: a frozen record of events and the final outcome. The agent cannot write persistent memory directly.

After the episode ends, the distiller converts supported observations into candidate memories. **Provenance** records where a memory came from, including the events supporting it. **Outcome polarity** distinguishes memories associated with success from those associated with failure. Both can contain useful information; a failed episode can reveal a constraint that a later agent should respect.

The embedder maps text to a numeric vector. Normalization makes vector lengths consistent for cosine comparisons. The exact index searches those vectors, while retrieval reserves separate places for positive and negative memories. The store remains authoritative: deleting the derived index must not destroy memories, and rebuilding it must restore the same exact-search results.

Consolidation chooses whether to insert a new item, merge it into a compatible prototype, or split an existing prototype. A **prototype** summarizes a group of memories using a representative vector and sufficient statistics such as the observation count and spread. Similar wording is not enough to justify a merge: polarity, type, scope, and contradiction checks protect distinctions that matter to behavior.

Finally, the compactor enforces the persistent-byte budget before the next generation starts. The next agent is constructed fresh. Its inheritance is the permitted contents of the memory store, not the previous agent's private state or the experiment logs.

## Gate requirements and current research status

The table identifies what must be demonstrated at each architecture gate. Module names locate responsibilities; their presence does not establish completion. The user has now approved the workshop task and frozen policy for G0–G1. Document 10 records their concrete checks; later evidence definitions and empirical comparisons remain open.

| Gate | Code responsibilities | Required evidence | Research status |
| --- | --- | --- | --- |
| G0: environment and determinism | `workshop_contract.py`, `workshop_environment.py`, `workshop_demo.py` | An approved Phase A generator and agent policy; exact evaluator ground truth; identical tasks, seeds, and actions reproduce observations and rewards. | Approved workshop implemented: all 4,096 mappings, transition boundaries, evaluator replay and separate-process determinism checked. See document 10. |
| G1: reset isolation | `agent.py`, `workshop_policy.py`, `workshop_demo.py`, retrieval interface | Fresh agents have no predecessor reference or writable store; private-state sentinels cannot cross generations through context, caches, globals, or logs. Repeat the experiment using the approved runtime. | Verified for the supplied trusted workshop policy: no inherited instance state, no hidden task/report inputs, prior policies released before successors. A future model adapter requires its own isolation checks; Python is not a hostile-code sandbox. |
| G2: insert-only memory loop | `distiller.py`, `embedder.py`, `memory_store.py`, `retrieval_policy.py` | Exactly one terminal write path; immutable trajectories; valid evidence references; normalized, revisioned representations; separate positive/negative retrieval quotas; no-memory retrieval remains empty. | Unvalidated on the approved task distribution. Research embedder and distillation rules still need specification. |
| G3: FIFO and hard budget | `pruning_policy.py`, `memory_store.py`, `vector_index.py`, `experiment_logger.py` | Actual charged bytes, including index overhead, fit B at every successful boundary; peak bytes are recorded; every eviction has a reason; infeasible coverage is explicit. | Budget invariants can be tested independently. Scientific budget and coverage definitions remain incomplete. |
| G4: consolidation | `consolidation_policy.py`, `memory_item.py` | Compatibility prevents invalid merges; novelty and projected distortion control decisions; statistics remain correct; Split has defined, budgeted evidence and verified behavior. | Incomplete: document 08 preserves one float32 centroid and explains the missing historical evidence needed for Split. Split semantics still need agreement and testing. |
| G5: inheritance | `experiment_runner.py`, isolated evaluation records | An approved task distribution supports transfer; matched-seed comparisons with no-memory across declared budgets; a predeclared statistical rule determines whether to continue. | Unvalidated. Stop and diagnose if the inheritance gate fails; do not proceed to utility experiments as though it passed. |
| G6: utility, H1 and H3 | `utility_estimator.py`, `utility_predictor.py`, retention policy | Paired rollouts produce utility estimates and standard errors; task-family separation prevents leakage; online predictions do not run rollouts; H1 and H3 receive held-out evaluations. | Unvalidated and dependent on G5. Prediction or statistics helpers alone do not establish either hypothesis. |
| G7: ANN scaling | `vector_index.py` behind the same interface | Approximate nearest-neighbor retrieval is measured against exact search; report recall, latency, rebuild cost and charged index bytes; verify downstream performance. | Later gate. Exact search remains the reference implementation. |
| G8: Phase B LLM transfer | Environment and agent adapters, generation runner | A selected LLM task environment, pinned models and prompts, frozen inference weights, complete cost records and repeated isolation/evaluation tests. | Later gate. No Phase B environment or model is selected by this guide. |

The build order limits how many unvalidated components enter an experiment at once. Helpers for a later gate may be developed and unit-tested early, but they must not be treated as permission to bypass an earlier research gate.

## H1: compare retention policies at equal persistent storage

H1 asks whether predicted downstream utility, information gain, and failure-family coverage improve retention compared with FIFO, random, and recency-based policies. A higher success rate with a larger memory is not sufficient evidence. The comparison must use the same task distribution, generation schedule, agent, retrieval rules, representation settings, and declared storage treatment, except for the condition being studied.

The primary study also calls for no-memory, raw-trajectory, reflection, and success-only conditions. These differ in more than eviction policy, so record their representations and selection rules explicitly. Their byte counts include the text, vectors, metadata, statistics, and index overhead that they retain. Report configured caps and actual charged bytes; equal caps alone need not produce identical realized storage. The procedure for achieving or interpreting equal-byte comparisons must be fixed before drawing H1 conclusions.

For each budget and condition, report success rate, repeated-failure rate, failure-family coverage, and the plan's efficiency measure:

`eta = (SR_policy - SR_nomemory) / MB_persistent`

Declare whether MB means 1,000,000 bytes and how storage varying across generations enters the denominator. The no-memory condition has zero persistent bytes, so its own efficiency ratio is undefined; retain its success rate as the reference and report efficiency as unavailable. Do not replace the denominator with an arbitrary nonzero number.

Record online memory-management cost, offline labeling cost, and persistent storage separately. An inexpensive online predictor can still require costly offline rollouts. Do not sum those costs into an unexplained single figure.

## H3: test whether local evidence and information gain predict utility

H3 predicts that measured information gain and local evidence sign carry more permutation importance than the source episode's terminal reward when predicting counterfactual memory utility. A model that accepts these features has not yet tested the hypothesis.

The target is a paired change in downstream return: evaluate the same task, seed, horizon, and agent policy with background memory M and with M plus candidate m. Save individual paired differences before computing their mean and standard error. Decide in advance whether adding m is allowed to exceed B during the counterfactual or must trigger compaction; the latter estimates a change that includes displacement of other memories. These are different estimands and must not be mixed without labels.

Freeze a candidate's input features at the time the online system would know them. The source episode's reward is available after that episode and may be a feature. Outcomes from the future rollouts used to label the candidate cannot be input features. Fit preprocessing and the predictor using training task families only; keep validation and test families separate. Also define how candidate origin families and rollout evaluation families are partitioned so shared evidence cannot cross the holdout boundary unnoticed.

Measure permutation importance on held-out families with a declared prediction metric and repeated permutations. Report uncertainty and the effect of correlated features. Under approved option 2, episode polarity retains its source-success/failure meaning for retrieval and merge compatibility. H3 instead uses the existing local `works` fact and measured IG as features against source reward. Binary source outcome is not an independent predictor of reward. Matched content and budgets ensure the comparison tests using a feature, not adding facts to only one condition. The IG estimator and statistical decision rules still need specification.

## Decisions needed before a research run

1. **Locate the complete plan.** The bibliography references sections 2, 28 and 29 of a numbered master plan that is not among the supplied project files. Its task definition or experimental rules may resolve the remaining choices. This guide does not replace that document.
2. **Apply the approved evidence distinction.** The workshop task/policy and option-2 source/local separation are approved. Document 11 records a terminal event extractor with source outcome, existing local `works` content and citations. It leaves IG unavailable and does not select a research embedder; the no-memory demo is not an inheritance comparison.
3. **Choose research representations.** Pin the embedder and revision, dimension, precision, text normalization, distillation rules and memory types. A deterministic local representation can help test the plumbing, but does not silently become the approved semantic embedder.
4. **Define information gain.** The bibliography describes a finite-hypothesis setting, but does not specify its hypotheses, prior, observation likelihood, posterior update, logarithm base or aggregation from events to memories. Provide these before using an information-gain feature as measured evidence. Do not substitute novelty or terminal reward for information gain.
5. **Define consolidation and Split.** Select the compatibility vocabulary, scope overlap rule, contradiction handling, novelty threshold, distortion statistic and ceiling. A centroid, count and scalar spread cannot reconstruct arbitrary original members. Document 08 proposes budgeted text replay for review while preserving one stored vector; no member-vector reservoir is proposed. Returning Insert where the architecture requires Split must not be presented as completed G4 behavior.
6. **Apply the resolved provenance rule.** Preserve immutable provenance while its memory exists, as the user directed. Charge it to B alongside the retained item. Evict the item's provenance when evicting that item; keep any separate audit history inaccessible to agents. A merged prototype must retain and charge its required source provenance; compact vectors do not authorize silently discarding source records.
7. **Make coverage feasible.** Define a failure family, rarity, minimum protected coverage, and behavior when protected records cannot fit. The system must not claim a successful boundary while violating B or its coverage contract. A clear infeasibility result is preferable to an undocumented relaxation.
8. **Fix byte accounting.** Specify serialization and framing, text encoding, vector precision, prototype evidence, index overhead, and treatment of fixed versus learned artifacts. In particular, a learned utility model carrying across generations needs an explicit storage classification. Rebuildable does not automatically mean free: the architecture charges index overhead to B.
9. **Predeclare experimental decisions.** Set budgets, task families, seeds, sample sizes, effect thresholds, confidence procedure, multiple-comparison treatment and the G5 stop/continue rule. A nonsignificant result alone does not establish equivalence. Address family and seed dependence when constructing intervals and testing differences.
10. **Fix H1 scoring and H3 evaluation.** Specify the retention formula, scaling of its features, coverage treatment, predictor selection procedure, utility rollout regime and held-out evaluation metric. Record every fitted artifact and prevent held-out results from choosing settings retrospectively.

## Where Python and C++ fit

Keep the first implementation mainly Python. The research logic is easier to inspect when configuration, trajectories, memory schemas, policies, experiment coordination, logging and analysis use the same readable language. No profiling evidence currently establishes that a custom C++ implementation is needed. The following are conditional migration candidates, based on their computational responsibilities rather than a claim of measured speedup.

| Component | Recommended initial home | Evidence that would justify native work |
| --- | --- | --- |
| Agent lifecycle, configuration, logging, experiment gates | Python | Keep these readable and close to the research definitions; numerical acceleration should happen behind their interfaces. |
| Retrieval scores and exact vector search | Python reference, then a measured library implementation | Large vector scans dominate online latency while policy and environment time are already understood. Preserve exact results and tie-breaking in reference tests. |
| G7 ANN index | Existing native-backed library with Python API | The exact-search baseline becomes too costly at the approved scale. Measure recall and charged index overhead alongside speed. |
| Prototype vector and distortion calculations | Python reference; batch operations before custom C++ | Profiling shows these numerical operations dominate consolidation. Preserve compatibility decisions in visible policy code. |
| Serialization and compaction loops | Python | Only migrate a measured hot loop; native and Python implementations must serialize the same schema and charge identical bytes. |
| Phase A transitions used by offline rollouts | Python until the task is settled | The approved simulator, rather than model inference or coordination, consumes a substantial share of labeling time. Replay parity is required. |
| Utility model fitting and statistical analysis | Python with appropriate libraries | Prefer existing numerical implementations; a custom native layer must not change labels, holdout rules or interpretation. |

For G7, Faiss already implements dense-vector similarity search in C++ and exposes Python wrappers. This makes a Python application with native search a concrete option without rewriting the project. Library selection remains a later, measured decision. See the [official Faiss documentation](https://faiss.ai/).

If a custom numerical kernel eventually earns its complexity, a narrow function accepting arrays and returning arrays is a more reviewable boundary than moving the full agent lifecycle into C++. pybind11 supports typed NumPy arrays and buffer interfaces; its documentation explains layout conversion and object-lifetime considerations. Validate array shape, precision, ownership and normalization at that boundary. See the [official pybind11 NumPy interface documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html).

Before a migration, capture the machine, runtime versions, workload, seed, item count, vector dimension, timing distribution and charged bytes. Afterward, run the same workload and compare both outputs and cost. For ANN, exact-neighbor parity is replaced by measured recall against the retained exact reference; downstream behavior must still be evaluated. Keep the Python reference available for correctness checks and simple reproduction.

## Later research and claims requiring verification

H4's quantization-versus-consolidation comparison and H5's typed-symbolic-versus-dense comparison remain in the plan as later research. They are not removed or considered satisfied by focusing current work on H1 and H3. H2's negative-memory question also remains part of the stated research scope, and its relationship to the success-only baseline should be retained in the experiment design.

The bibliography explicitly leaves several entries unverified, including CER, adaptive budgeted forgetting, some identifiers and the motivating incident. Verify those against primary sources before citing them. The adopted framing acknowledges byte-budgeted memory work such as [WritePolicyBench](https://arxiv.org/html/2602.02574v1) and [Forget to Improve / CURATOR](https://arxiv.org/html/2606.25115v1). GIM's proposed contribution concerns successor-agent outcomes and utility attribution under audited storage limits. Its novelty must be assessed against the relevant prior work; implementing the experiment does not establish a first-of-its-kind claim.

## What a reproducible result must contain

A research run should save the configuration, task generator and family identifiers, seeds, runtime and dependency versions, model and prompt revisions, representation settings, serialization version and fitted-artifact identifiers. Its analysis records should contain paired outcomes, actual and peak byte counts, retrieval decisions, consolidation and eviction reasons, and separately measured online and offline costs. These records must remain inaccessible to successor agents.

Save a gate report that distinguishes invariant-test results from empirical results. Include the approved statistical decision rule, uncertainty estimates, failures and the reason for any stop. A reproducible smoke test demonstrates that another person can run the implementation; a reproducible research result additionally requires the approved scientific definitions and evidence described above.
