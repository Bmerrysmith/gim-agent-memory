# UTILITY PREDICTOR - PSEUDOCODE ONLY - G6
# OFFLINE:
#     EXTRACT geometry, provenance, local evidence, and information-gain features
#     using only information that would exist when an online decision is made.
#     KEEP source outcome as the architecture's retrieval/merge label.
#     READ local evidence sign from existing typed content (workshop: works).
#     COMPARE local evidence and measured IG with actual source episode reward.
#     DO NOT treat the binary source-outcome label as an independent reward signal.
#     HOLD memory content and byte budgets equal across feature comparisons;
#     the test is whether using the feature helps, not whether new facts were added.
#     SPLIT by task family before fitting; prevent related-episode leakage.
#     FIT on offline utility labels and evaluate on held-out families.
#     SAVE a versioned artifact with feature schema and uncertainty diagnostics.
# ONLINE:
#     LOAD the fixed artifact and validate feature compatibility.
#     PREDICT utility at fixed cost per memory without running rollouts.
#     PASS predicted scores to the configured compactor policy.
# DECLARE treatment of fixed model-artifact bytes consistently across experiments;
# any artifact updated across generations must not become unaccounted memory.
