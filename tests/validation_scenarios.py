# VALIDATION SCENARIOS - PSEUDOCODE ONLY; THESE ARE NOT EXECUTABLE TESTS
# G0: same task, seed, and action sequence reproduces observations and rewards.
# G1: plant a transient-state sentinel in generation n; verify it cannot reach
# generation n+1 through objects, context, caches, globals, or experiment logs.
# G2: terminal write runs once; every candidate cites valid frozen event IDs;
# the agent cannot mutate the store; no-memory retrieval stays empty.
# G3: exercise empty store, zero budget, oversized item, index growth, and eviction;
# compare accounted bytes with serialization and reject infeasible coverage rules.
# G4: near-identical success/failure texts cannot merge; compatible items may merge;
# excessive distortion splits only with sufficient retained evidence.
# G4: reconstruct the index from the store and compare exact-search results.
# G5: compare matched-seed memory conditions with the no-memory control and report
# uncertainty before deciding whether inheritance justifies the next gate.
# G6: paired utility labels match seed/task/horizon; held-out families never leak
# into training; online predictions never invoke the offline rollout evaluator.
# G7: approximate-search quality is measured against the exact-search reference.
# G8: frozen model and embedding revisions are recorded and reset isolation holds.
# ALL: analysis logs cannot become an unbudgeted read path for later agents.
