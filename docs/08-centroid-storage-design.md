# Compact centroid storage and the remaining Split decision

Status: revised proposal after the user's preference to keep centroid-vector bytes.
The four-group Split proposal in document 06 is withdrawn. This document supersedes
that proposal's storage and Split sections. The user subsequently approved the
Phase A environment/policy for G0–G1 (document 10) and option-2 source/local evidence
separation (document 11). The IG estimator, utility estimand and a replacement for
Split remain unfinished. The architecture document remains unchanged.

The priorities are small persistent storage, useful experience transferred to
successor agents, lower deployed token use, and measured speed. Compact vectors
serve the first priority; usefulness, token savings and speed still need evidence.

Choose among designs by measuring task success and repeated failures alongside
bytes, total deployed tokens and latency. Prefer a cheaper design when useful
transfer is preserved; otherwise show the measured quality/cost tradeoff rather
than inventing a conversion between bytes, tokens and milliseconds. H1's equal-byte
comparison remains intact. A minimum acceptable quality level is an experimental
decision, not assumed from this preference.

## Keep the architecture's vector allocation

Persist **one dimension-d float32 centroid per memory/prototype: exactly 4d bytes**.
At d = 384, that is **1,536 vector bytes**, matching architecture section 6.
Do not persist representative embeddings, microcluster vectors, float64 vector
sums, member-vector archives, or per-coordinate compensation arrays.

Keep the geometric statistics scalar: a member count and a spread statistic, with an
optional squared-norm sum and conservative radius bound. For example, a uint64
count and three float64 statistics cost 32 bytes total. This is an illustrative
scalar layout, not an approved new distortion algorithm or a claim that the
complete prototype occupies only 1,568 bytes.

Content, scope, learned fields, immutable provenance, serialization framing and
the index's charged representation also count toward B. Preserve every retained
memory's provenance while that memory exists, as the user specified. Merging
does not make accumulated provenance free; its growth can eventually require
eviction of the whole prototype. Eviction removes the prototype and its attached
provenance together. Separate audit logs cannot become an unbudgeted memory source.

The authoritative serialized vector must actually be float32 bytes. Merely
declaring 4d while saving float64 coordinates or JSON numeric arrays would charge
a different representation than the one retained. Temporary calculation arrays
do not add persistent vectors; report runtime costs separately.

## Mean, retrieval direction, and numerical precision

For the proposed future prototype, the centroid represents the **arithmetic mean**
of its member vectors. A basic unmerged item instead keeps its normalized
embedding rounded to float32; the compact codec does not implement prototypes.
Do not renormalize and overwrite that mean after a merge. Its length carries
information about dispersion. Cosine retrieval may derive a unit direction
transiently from the stored centroid; any retained index copy is charged under
the architecture's index accounting.

For an existing count n, mean c and new embedded vector x, the ideal update is:

`new_mean = c + (x - c) / (n + 1)`

Calculate with wider temporary arithmetic, then round the persisted centroid to
float32. Once c has been rounded, the update is an approximation to the historical
mean. Fixed inputs and merge order can reproduce that approximation; reproducible
does not mean mathematically exact or independent of merge order. Compatibility
checks remain mandatory before averaging any vectors.

If Q is the sum of squared input norms and mu is their true arithmetic mean,
the historical mean squared spread is `Q / n - ||mu||^2`. Substituting a rounded
stored centroid gives an approximation. Float32 versions of normalized embeddings
also need not have squared norm exactly one, so do not silently substitute `Q = n`.
Scalar Welford-style updates likewise accumulate error when their mean is rounded.
The distortion metric, negative-roundoff tolerance and threshold policy require
an explicit specification and verification before claiming G4; this revision
does not approve the earlier RMS choice.

A zero centroid has no cosine direction, even when all member embeddings are
nonzero. The retrieval/consolidation contract must handle it explicitly instead
of dividing by zero or inventing an uncharged representative direction. Treating
it as an invalid prototype is possible; choosing its operational fallback remains
part of the unfinished consolidation design.

## Radius is a bound, not an exact maximum

An exact maximum distance to an old centroid cannot generally be updated exactly
after that centroid moves using only the old maximum and the new sample. A
conservative radius bound can be updated without retaining historical vectors.
If the old stored center is a, its valid bound is r, the newly stored float32
center is b, and the new member is x, use the geometric bound:

`new_radius_bound >= max(r + ||a - b||, ||x - b||)`

This follows from the triangle inequality. The bound is around the **stored**
center, so mean rounding is included in its movement. An implementation claiming
a rigorous upper bound must also round distance/arithmetic results outward or
provide a justified error allowance. Name it `radius_bound`; do not report it as
an exact `max_dist`. It can grow conservatively and cause earlier rejection.

## What these statistics cannot recover

Centroid, count, spread and radius can summarize a merged population. They cannot
reconstruct historical cluster directions, memberships, or child centroids.
For example, `(a, +b, 0)` and `(a, -b, 0)` have the same mean, count, spread,
norm sum and radius as `(a, 0, +b)` and `(a, 0, -b)`, with `a^2 + b^2 = 1`.
The two populations require different historical splits. The retained statistics
cannot distinguish them. Source references preserve attribution, not discarded
vector geometry; reading an unbudgeted archive to recover it would change the
memory experiment.

Keeping the old prototype and separately inserting a candidate whose projected
merge would exceed the distortion ceiling is **Insert**, not a recovered Split of that
prototype. It may be useful behavior, but adopting it for the architecture's
`projected distortion > ceiling -> Split(target)` branch changes that branch.
The user's storage preference alone does not authorize that change.

## Alternative for review: replay budgeted member text during Split

Retain every unique member's exact embedding input text, multiplicity and full
immutable provenance in the budgeted prototype metadata. Preserve every other
input field needed by the encoder and the provenance-to-input association; a
representative summary cannot replace the original inputs. Repeated identical
inputs share one text entry and count. Persist only one centroid, with no member
vectors or vector sums. This extra text supplies the information missing from
pure centroid/scalar statistics and can make a real partition of past members possible.

When Split is needed, re-embed those inputs using the pinned deterministic encoder
and identical preprocessing, normalization and float32 conversion. The reconstructed
vectors must reproduce the originals; a pinned model revision alone is not proof
of this. Recommend a charged digest of the original canonical float32 vector per
unique encoder input to detect mismatch. Fail the operation on mismatch; do not
silently split using changed embeddings. Exact replay is a condition to verify,
not a guarantee supplied by arbitrary LLM embedding services.

An approved deterministic partition can then assign historical inputs, counts and
their provenance to children, recompute scalar statistics and one centroid per
child, and select a retained text as each child's representative. This partitions
actual historical members; it does not recover unknown historical cluster labels
or promise an optimal clustering. Discard reconstructed vectors when the write
finishes, at latest by the generation boundary. Report temporary peak RAM and
re-embedding time as online write-path costs; enforce B after the ordinary boundary
compaction. Never recover inputs from the unbudgeted logger.

All growing text, counts, input references, digests and provenance count toward B.
This explicitly extends the sketch's approximately 200-byte content allowance and
requires approval; savings versus vector reservoirs depend on actual text lengths.
Before a split, per-step search still uses one centroid and the agent receives
only representative content, scope and required evidence, not the member archive.
The archive need not add prompt tokens or vector-search work, but storage loading,
accounting and occasional Split become more expensive. Measure these costs.

## Useful experience, token cost, and speed

Repeated compatible evidence can reinforce a memory's support count and append
its immutable provenance without storing another vector. Count is evidence of
support, not proof of downstream utility. H1 still needs equal-byte retention
comparisons. H3 now uses the approved local `works` fact separately from source
outcome; its information-gain estimator still needs specification. Do not discard
provenance or invent an IG measurement to obtain a smaller record.

Here reinforcement means better-supported external memory used by a fresh agent.
The agent's inference weights remain frozen as required by the architecture;
this proposal does not introduce reinforcement-learning weight updates.

Vector bytes and model tokens measure different costs. At G8, measure retrieved
memory text with the deployed model's actual tokenizer, record prompt/completion
usage, and compare useful transfer against a declared token allowance. Select
whole relevant memories under a reviewed allowance; do not silently truncate away
scope, negative evidence or other meaning. Character-to-token estimates are not
evidence of savings. This is a measurement recommendation for G8, not approval to
skip the preceding gates or redefine B as a token budget.

Keep Python for the current implementation. Measure serialization, consolidation
and retrieval latency separately at realistic store sizes before considering
C++. Compiled geometry or search kernels may help if profiling identifies them
as bottlenecks; the current storage preference alone proves no speed advantage.
Do not add speculative caches that increase persistent bytes without accounting.
Use profiling to locate hotspots and separate benchmarks to compare execution
times; profiling can bias Python-versus-C comparisons. See the
[Python profiling documentation](https://docs.python.org/3/library/profile.html).

## Next decision

**Keep 4d vector bytes; review budgeted text replay if historical Split is required.**
That option preserves a real Split without a persistent vector reservoir, at the
cost of retained text and occasional recomputation. Approving it still leaves
the partition algorithm, distortion contract and encoder replay verification to
specify before G4. It is proposed here, not implemented or already approved.

If additional retained text is unacceptable, defer Split at G4. The approved
environment now has G0–G1 checks; G2 still needs its evidence definitions and
embedder. Do not claim G4 or a validated G5/G6 experiment. A third choice is an explicit architecture
amendment replacing the high-distortion branch with Insert while preserving the
old prototype. That changes the branch and its gate; do not label it Split.
