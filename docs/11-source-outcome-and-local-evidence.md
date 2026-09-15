# Option 2: source outcome and local evidence remain separate

Approved by the user on 2026-09-14. This replaces the earlier proposal to redefine
memory polarity as an individual tool result. The architecture's episode-based
retrieval groups and source-outcome merge restriction retain their meaning.

## The two meanings

`outcome_class` answers: **did the whole source episode succeed?** In the workshop,
success means all three locks opened within five attempts. The actual terminal
reward stays in the memory's immutable provenance.

The existing content field `works` answers: **did this particular tool open this
particular lock?** It is scoped evidence, not a measured utility score. It does not
say how much the fact helps a future agent.

| Source episode | Local `works` | Retrieval group | What the memory says |
| --- | --- | --- | --- |
| Success | true | `k+`: source success | This tool opened this lock. |
| Success | false | `k+`: source success | This attempt failed before the episode recovered. |
| Failure | true | `k-`: source failure | This tool opened this lock, but the episode later failed. |
| Failure | false | `k-`: source failure | This attempt failed during a failed episode. |

All four cases retain their citations. No local fact is discarded merely because
it differs from its source episode's final result. A future merge must preserve
source-outcome compatibility as well as type, scope and contradiction rules.
Same-source labels alone never justify merging opposite local assertions.

## H3 amendment

H3 now compares **measured information gain and local evidence sign** against
source episode reward when predicting counterfactual memory utility. The H3
paragraph and matching abstract sentence have been updated explicitly; the
architecture diagram and other hypotheses retain their definitions.

Source polarity and binary source reward contain the same information in this
workshop, so the predictor must not treat them as independent signals. Derive the
local feature from the existing `works` content; do not add another stored flag
or another centroid to represent it.

Every comparison condition keeps the same memory content and byte budget. The
question is whether using the local fact as a feature improves prediction or
retention. Test on held-out workshop families and report uncertainty and feature
correlation. Permutation importance does not establish causality. Predictor and
utility experiments remain at G6 after the inheritance gate.

## Implemented terminal extraction

`src/gim/workshop_distiller.py` supplies `WorkshopDistiller`. Given a complete
frozen workshop trajectory, it validates the public sequence and emits one
`CandidateMemory` per observed attempt:

- Type `workshop-tool-rule-v1`; exact workshop scope.
- Canonical content containing only `lock_type`, `tool`, and `works`.
- The whole episode's `outcome_class`, with the attempt's event citation.
- `information_gain=None`, since this extractor does not compute an IG estimator.
- A `failure_family` tag for every observed wrong-tool result, irrespective of
  the whole episode's outcome. The tag is the observed family/lock/tool triple
  encoded as compact JSON in the existing field.

The ordinary runner attaches the actual source reward, embeds the candidate and
charges its full serialized bytes. Extraction never reads hidden rules or audit
logs. It rejects internally inconsistent evidence; a coherent fabricated transcript
cannot be authenticated without a trusted source. The separate environment
evaluator checks genuine experiment trajectories against hidden ground truth.

Generic memory validation now permits a source-success record to carry a local
failure-family tag. This preserves coverage for an observed warning without
moving the memory into the source-failure quota. Coverage selection still requires
explicit named families; no rarity definition or scientific quota is invented.

## Cost and remaining work

The decision adds no vector or duplicate local-sign field. A 384-dimensional
vector remains 1,536 bytes. Actual content, provenance and failure-family tags
are still charged metadata. No deployed token reduction or speed gain is claimed
from this semantic decision.

The distiller and source-based retrieval interaction are checked with explicit
fixture embeddings. They do not choose the research embedder or supply a full G2
study. The existing workshop demo remains a G0–G1 no-memory run. The IG estimator,
research representation, G4 Split evidence and later statistical protocol remain
separate unfinished work.
