# GIM — Architecture Diagrams

## 1. System overview

```mermaid
flowchart TB
    subgraph GEN["One generation (destroyed at boundary)"]
        AG["Agent<br/>frozen weights, no store write access"]
        ENV["Environment<br/>ground_truth() in Phase A only"]
        AG -- action --> ENV
        ENV -- observation --> AG
        AG -- typed events --> TRAJ["Trajectory<br/>event list + terminal state + reward"]
    end

    subgraph WRITE["Write path — terminal, once per episode"]
        DIST["Distiller<br/>every candidate cites event ids"]
        EMB["Embedder<br/>L2-normalized, pinned revision"]
        CONS["ConsolidationPolicy<br/>Insert | MergeInto | Split"]
        DIST --> EMB --> CONS
    end

    subgraph PERSIST["Persistent memory — bounded by B"]
        STORE[("MemoryStore<br/>items + prototypes<br/>exact byte accounting")]
        IDX["VectorIndex<br/>derived cache, rebuildable"]
        STORE -. rebuild .-> IDX
    end

    subgraph READ["Read path — per step"]
        RET["RetrievalPolicy<br/>separate k+ / k- quotas"]
    end

    PRUNE["PruningPolicy / compactor<br/>runs at generation boundary<br/>sole owner of B"]

    LOGS[("ExperimentLogger<br/>episode / memory_event / retrieval<br/>UNBUDGETED, separate store")]

    TRAJ --> DIST
    CONS --> STORE
    CONS --> IDX
    STORE --> PRUNE
    PRUNE -- evictions --> STORE
    STORE --> RET
    IDX --> RET
    RET -- retrieved memories --> AG
    CONS --> LOGS
    PRUNE --> LOGS
    RET --> LOGS
    TRAJ --> LOGS

    style PERSIST fill:#e3ece8
    style LOGS fill:#f0f0f0
```

Read the diagram for three things. The agent has no arrow into the store — only the compactor and consolidation write. The index hangs off the store with a dashed rebuild edge, because it is derived and never authoritative. The logger sits outside the budgeted box.

---

## 2. The generation boundary

```mermaid
sequenceDiagram
    participant R as GenerationRunner
    participant A as Agent (gen n)
    participant E as Environment
    participant S as MemoryStore
    participant C as Compactor

    R->>A: construct fresh (policy, retrieval handle)
    loop each step
        A->>S: retrieve(query)  [read only]
        S-->>A: k+ positive, k- negative
        A->>E: action
        E-->>A: observation, reward, done
    end
    A-->>R: Trajectory (frozen)
    R->>R: distill -> embed -> consolidate
    R->>S: insert / merge / split
    Note over S: peak_bytes recorded here
    R->>C: enforce(B)
    C->>S: evictions (each logged with reason)
    Note over S: assert bytes() <= B
    R->>A: destroy
    Note over R,A: no object identity crosses this line
```

---

## 3. Consolidation decision

```mermaid
flowchart TD
    C["candidate memory + vector"] --> NN["index.search(k=m)<br/>filtered to compatible items"]
    NN --> E0{"any compatible<br/>neighbor?"}
    E0 -- no --> INS1["Insert<br/>reason: no compatible neighbor"]
    E0 -- yes --> NOV{"novelty N = 1 - sim<br/>&gt; threshold?"}
    NOV -- yes --> INS2["Insert<br/>reason: novel"]
    NOV -- no --> COMPAT{"hard compatibility:<br/>same outcome_class,<br/>same memory_type,<br/>overlapping scope,<br/>no contradiction"}
    COMPAT -- fails --> INS3["Insert<br/>reason: similar but incompatible"]
    COMPAT -- passes --> DIST{"projected distortion<br/>&lt;= ceiling?"}
    DIST -- yes --> MERGE["MergeInto(target)<br/>update centroid + n_k + spread"]
    DIST -- no --> SPLIT["Split(target)"]

    style INS3 fill:#f5e6e6
    style MERGE fill:#e3ece8
```

The "similar but incompatible" branch is the one the whole design exists to protect. A success memory and a failure memory that are textually near-identical must never be averaged together, and cosine similarity alone cannot see the difference.

---

## 4. Utility pipeline — offline vs online

```mermaid
flowchart LR
    subgraph OFF["OFFLINE — batch, expensive, O(n·R·H)"]
        RO["paired-seed rollouts<br/>G(M ∪ m) - G(M)"] --> LAB["U_hat labels<br/>with standard errors"]
        LAB --> FEAT["feature extraction<br/>no post-outcome leakage"]
        FEAT --> FIT["fit utility predictor<br/>split by task family"]
        FIT --> MODEL[("model artifact")]
    end

    subgraph ON["ONLINE — cheap, fixed cost per memory"]
        NEW["new memory"] --> FX["features"] --> PRED["predicted utility"]
        PRED --> SCORE["retention score"]
        SCORE --> COMP["compactor"]
    end

    MODEL -.-> PRED

    style OFF fill:#f0f0f0
```

The dashed edge is the only coupling. Nothing in the online path calls a rollout. Report the two costs separately and never sum them.

---

## 5. Build order and gates

```mermaid
flowchart LR
    G0["G0<br/>env + determinism"] --> G1["G1<br/>reset isolation"]
    G1 --> G2["G2<br/>memory loop<br/>insert only"]
    G2 --> G3["G3<br/>budget enforced<br/>FIFO"]
    G3 --> G4["G4<br/>consolidation<br/>prototypes"]
    G4 --> G5{"G5<br/>INHERITANCE GATE<br/>does any memory beat<br/>no-memory at equal bytes?"}
    G5 -- no --> STOP["stop and diagnose<br/>kill test fired"]
    G5 -- yes --> G6["G6<br/>utility<br/>MC + predictor"]
    G6 --> G7["G7<br/>scaling<br/>ANN index"]
    G7 --> G8["G8<br/>LLM transfer<br/>Phase B"]

    style G5 fill:#f5e6e6
    style STOP fill:#f5e6e6
```

Exactly one novel component is unvalidated at any point. That ordering is the main defense against an unfinished project.

---

## 6. Byte budget composition

```
persistent bytes, per item
+-------------------------------------------------------------+
| content text        ~200 B    distilled memory               |
| centroid vector     ~1536 B   384-dim float32   <-- dominant |
| preconditions/scope ~100 B    json                           |
| sufficient stats    ~40 B     n_k, spread, max_dist          |
| provenance          ~150 B    immutable, never evicted       |
| learned fields      ~40 B     utility, confidence, info gain |
+-------------------------------------------------------------+
plus index.overhead_bytes(), which counts toward B

Roughly 75-90% of every memory is the embedding. That single fact is
what makes the symbolic-vs-dense comparison (H5) worth running, and
what makes quantization vs consolidation (H4) a real question rather
than a detail.
```
