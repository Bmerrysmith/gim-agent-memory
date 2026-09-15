# OFFLINE UTILITY ESTIMATOR - PSEUDOCODE ONLY - G6
# FOR each candidate memory and evaluation context:
#     RUN paired-seed rollouts with memory M and with M plus the candidate.
#     KEEP tasks, seeds, horizon, and agent policy equal within each pair.
#     MEASURE the paired change in downstream return.
#     AGGREGATE differences into a utility estimate with standard error.
#     LABEL the evaluation regime and account for storage/compaction effects.
# STORE labels in the offline experiment data, inaccessible to online agents.
# TRACK rollout cost separately from online memory management.
# AVOID assuming that a positive source reward implies positive marginal utility.
