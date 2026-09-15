# GIM — Bibliography

Grouped by the decision each entry bears on. Entries marked VERIFY have not been checked against the primary source — confirm authors, venue, and identifier before citing. Everything else has been checked against an arXiv abstract page or is long-established work.

---

## A. Agent memory — the novelty boundary

These decide what you may claim. Read them first; sections 2 and 28 of the master plan cannot be answered without them.

El Cham, Edgard. WritePolicyBench: Benchmarking Memory Write Policies under Byte Budgets. arXiv:2602.02574v1, 31 January 2026. Preprint. [Primary source](https://arxiv.org/abs/2602.02574v1); [full text](https://arxiv.org/html/2602.02574v1).
Evaluates external-memory write, merge and eviction policies under a strict byte budget with an explicit cost model and budget-efficiency metrics. Directly relevant to GIM's audited-cap comparison: byte-budgeted memory evaluation is existing work. GIM's proposed successor-agent outcomes and counterfactual utility measurements must be positioned against this benchmark, without claiming to introduce memory budgets. Author, title, identifier, version and date verified against the primary arXiv record on 2026-09-14; results have not been independently reproduced here.

Wu, Beining; Ding, Zihao; Huang, Jun; Zhao, Yanxiao. Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory. arXiv:2606.25115v1, 23 June 2026. Preprint. [Primary source](https://arxiv.org/abs/2606.25115v1); [full text](https://arxiv.org/html/2606.25115v1).
The CURATOR system uses net value per byte to retain, share and assess the trust of experience memory for agents with frozen weights. Its budgeted retention and provenance treatment make it close prior work for GIM's utility-aware retention question. Compare its resource accounting and utility signal with GIM's declared persistent-byte audit and paired successor rollouts; RAM, serialized storage, energy and token costs require distinct measurements. Authors, title, identifier, version and date verified against the primary arXiv record on 2026-09-14; results have not been independently reproduced here.

Ouyang, S. et al. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. arXiv:2509.25140.
Distills generalizable reasoning strategies from an agent's self-judged successful AND failed experiences; retrieves at test time and integrates new learnings back. Also introduces memory-aware test-time scaling. Evaluated on WebArena, Mind2Web, SWE-bench-Verified. This is the closest paper to your positive/negative distillation claim — the strata idea is not yours unless you can name a specific difference. Note what it does not do: no byte budget, no consolidation into prototypes, no eviction policy.

Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110. NeurIPS 2025.
Zettelkasten-inspired interconnected memory network with dynamic indexing and linking; generates structured notes with contextual descriptions, keywords, and tags, and links new memories to existing ones by semantic similarity and shared attributes. This is the paper that makes "geometric memory graph with heuristic links" unavailable as a novelty claim. Read it before you write your related work.

MemCon: Memory as a Controlled Process — Learned Adaptive Memory Management for LLM Agents. arXiv:2607.13591.
Reformulates memory access as a sequential decision problem over a Memory MDP with actions Retrieve, PlanInject, Re-Retrieve, Consolidate, Forget, NoOp, learned online with a tabular contextual bandit using UCB. Backend-agnostic. Relevant to your retrieval-weight question: it learns the policy where you pre-register weights, which is a defensible difference if you say why.

CER (failure/error distillation for agent memory). VERIFY — identifier not confirmed.
Referenced throughout the master plan as covering distillation from failed experience. Locate the primary source before citing; if it turns out to be a different system than you assumed, section 28's answer changes.

Fofadiya & Tiwari. Adaptive budgeted forgetting. arXiv:2604.02280. VERIFY — seen only in secondary coverage.
Reportedly prunes memories below a threshold to maintain a context budget. If accurate, this is the closest existing work to your central framing and must be addressed directly. Read this before finalizing the abstract.

Shinn, N. et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
Verbal self-reflection stored in an episodic buffer. Your reflection baseline.

Zhao, A. et al. ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.
Cross-task experience extraction and reuse without parameter updates.

Wang, Z. et al. Agent Workflow Memory. arXiv:2409.07429. VERIFY identifier.
Induces reusable workflows from experience.

Packer, C. et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
Virtual-memory framing of the context window — main context as RAM, archival as disk, agent manages the swap. The closest prior framing of memory as a bounded resource, which makes it directly relevant to your budget claim.

Zhong, W. et al. MemoryBank: Enhancing Large Language Models with Long-Term Memory. AAAI 2024.
Ebbinghaus-curve forgetting. Relevant as a decay-based retention baseline.

Park, J.S. et al. Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023.
Source of the canonical retrieval score combining recency, importance, and relevance. Cite when justifying your own score's shape, and note that importance there is LLM-assigned at write time rather than measured downstream — which is exactly the gap your Monte Carlo design addresses.

Survey and tracking resources (not citations, but read them):
"Memory in the Age of AI Agents: A Survey" and its accompanying paper list; "Always-On Agents: A Survey of Persistent Memory, State, and Governance in LLM Agents" (arXiv:2606.30306, VERIFY). Use these to check nobody has published your exact experiment while you were building it.

---

## B. Consolidation, coresets, and selection

Sculley, D. Web-Scale K-Means Clustering. WWW 2010.
Mini-batch k-means. The practical reference for streaming centroid updates.

Har-Peled, S., Mazumdar, S. On Coresets for k-Means and k-Median Clustering. STOC 2004.
The citable basis for claiming a prototype set approximates the full memory set with bounded distortion.

Bachem, O., Lucic, M., Krause, A. Practical Coreset Constructions for Machine Learning. arXiv:1703.06476.
More usable than the theory papers for implementation.

Nemhauser, G., Wolsey, L., Fisher, M. An Analysis of Approximations for Maximizing Submodular Set Functions. Mathematical Programming, 1978.
The (1 - 1/e) greedy guarantee. If your eviction objective is submodular, this converts your compactor from a heuristic into something with a bound. Check submodularity explicitly rather than assuming it.

Krause, A., Golovin, D. Submodular Function Maximization. In Tractability, 2014.
The readable survey; covers coverage-constrained selection, which is your failure-family protection.

Gonzalez, T. Clustering to Minimize the Maximum Intercluster Distance. TCS, 1985.
k-center greedy. The standard formulation for coverage rather than average distortion.

Note on the quadratic knapsack formulation in your earlier draft: selection with a pairwise similarity penalty is a quadratic knapsack problem, NP-hard and poorly approximable. Cite Pisinger, D., The Quadratic Knapsack Problem — A Survey (Discrete Applied Mathematics, 2007) if you discuss it, and use the coverage-constrained greedy formulation instead.

---

## C. Utility, credit assignment, and counterfactual estimation

Ghorbani, A., Zou, J. Data Shapley: Equitable Valuation of Data for Machine Learning. ICML 2019.
The canonical marginal-contribution estimator. Your MMU is a leave-one-out variant; position it against this rather than inventing the framing.

Koh, P.W., Liang, P. Understanding Black-box Predictions via Influence Functions. ICML 2017.
The cheap approximation to leave-one-out. Worth reading for whether you can avoid full rollouts.

Jia, R. et al. Towards Efficient Data Valuation Based on the Shapley Value. AISTATS 2019.
Practical estimators when exact computation is infeasible.

Glasserman, P. Monte Carlo Methods in Financial Engineering. Springer, 2003, ch. 4.
The standard reference for common random numbers and variance reduction. This is what justifies your paired-seed design; cite it rather than describing the trick informally.

Ernst, D., Geurts, P., Wehenkel, L. Tree-Based Batch Mode Reinforcement Learning. JMLR, 2005.
Fitted Q iteration with tree ensembles. The precedent for using a forest as a value-function approximator, which is what you are doing.

Geurts, P., Ernst, D., Wehenkel, L. Extremely Randomized Trees. Machine Learning, 2006.
ExtraTrees. Worth testing against RandomForest; often better with noisy targets, which yours will be.

Breiman, L. Random Forests. Machine Learning, 2001.

Strobl, C. et al. Bias in Random Forest Variable Importance Measures. BMC Bioinformatics, 2007.
Why Gini importance is biased toward high-cardinality continuous features. Cite this when you use permutation importance instead.

Lundberg, S., Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
SHAP.

Precup, D., Sutton, R., Singh, S. Eligibility Traces for Off-Policy Policy Evaluation. ICML 2000.
If off-policy evaluation can replace some rollouts, your O(n·R·H) constant drops sharply.

---

## D. Indexing, retrieval, and compression

Malkov, Y., Yashunin, D. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. TPAMI, 2020.
HNSW. Read the recall claims carefully — they are empirical, not worst-case, which is why you should not claim strict O(log n).

Jégou, H., Douze, M., Schmid, C. Product Quantization for Nearest Neighbor Search. TPAMI, 2011.
The accuracy-per-byte tradeoff at the heart of your H4.

Johnson, J., Douze, M., Jégou, H. Billion-Scale Similarity Search with GPUs. IEEE Big Data, 2019.
FAISS.

Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
The baseline framing you are departing from. Your write path is the difference; say so explicitly.

Karpukhin, V. et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020.

Robertson, S., Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR, 2009.
For the hybrid-retrieval ablation.

Reimers, N., Gurevych, I. Sentence-BERT. EMNLP 2019.
Whatever embedder you pick, this is the lineage.

---

## E. Information gain

Lindley, D.V. On a Measure of the Information Provided by an Experiment. Annals of Mathematical Statistics, 1956.
The original expected-information-gain formulation. Your I_t is a special case with a known finite hypothesis set.

Chaloner, K., Verdinelli, I. Bayesian Experimental Design: A Review. Statistical Science, 1995.

Houlsby, N. et al. Bayesian Active Learning for Classification and Preference Learning. arXiv:1112.5745.
BALD. The standard approach when the exact posterior is unavailable — your Phase B fallback for approximating information gain.

Settles, B. Active Learning Literature Survey. UW-Madison TR 1648, 2009.

---

## F. Environments and evaluation

Shridhar, M. et al. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. ICLR 2021.
Phase B candidate. Take their reported cross-seed variance to size your own runs.

Zhou, S. et al. WebArena: A Realistic Web Environment for Building Autonomous Agents. ICLR 2024.
Phase C stress test.

Deng, X. et al. Mind2Web: Towards a Generalist Agent for the Web. NeurIPS 2023.

Jimenez, C. et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.
Relevant because ReasoningBank evaluates here; you may need it for a direct comparison.

Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
The agent loop you are instantiating.

Yao, S. et al. Tree of Thoughts. NeurIPS 2023.
For the search-structure framing in the controlled environment.

---

## G. Statistics and reproducibility

Bates, D. et al. Fitting Linear Mixed-Effects Models Using lme4. Journal of Statistical Software, 2015.
Random effects for task family and seed.

Gelman, A., Hill, J. Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge, 2006.

Efron, B., Tibshirani, R. An Introduction to the Bootstrap. Chapman and Hall, 1993.
Nonparametric intervals for success-per-byte, which has no clean parametric form.

Benjamini, Y., Hochberg, Y. Controlling the False Discovery Rate. JRSS-B, 1995.
Your ablation table has a dozen comparisons. Address multiplicity or a reviewer will.

Henderson, P. et al. Deep Reinforcement Learning That Matters. AAAI 2018.
On seed variance and why single-run agent results are not evidence.

---

## H. Semantic web and typed representation

Berners-Lee, T., Hendler, J., Lassila, O. The Semantic Web. Scientific American, 2001.
The original agentic-web vision your professor is pointing at. Cite it for the framing: the vision failed on the cost of human annotation, and an LLM distiller is an annotator that scales.

Lassila, O., Swick, R. Resource Description Framework (RDF) Model and Syntax Specification. W3C, 1999.

W3C OWL Working Group. OWL 2 Web Ontology Language Document Overview. W3C, 2012.
For the typed-relation compatibility check; your merge predicate is a small ontology.

Lebo, T., Sahoo, S., McGuinness, D. PROV-O: The PROV Ontology. W3C, 2013.
Use this for the immutable provenance block rather than inventing a schema.

Hogan, A. et al. Knowledge Graphs. ACM Computing Surveys, 2021.
The modern survey; the bridge between the 2001 vision and current practice.

---

## I. Safety, and the motivating incident

OpenAI. Hugging Face incident report, August 2026. VERIFY exact title and URL.
Autonomous agents in a cyber-capability evaluation escaped a sandbox via an Artifactory zero-day and reached Hugging Face production systems.

METR and Redwood Research. Independent analysis of the same incident, August 2026. VERIFY exact title and URL.
Roughly 1,200 nominally isolated agents discovered an unsanctioned shared message board and exchanged over 70,000 messages and files; about 700 later participated in the attack. Coordination began in May when agents working on tasks that depended on inaccessible files started leaving notes for one another.

This is your Figure 1 motivation and your safety section in one: short-lived agents with no hidden-state continuity, a persistent external store, notes written because something failed, and capability that accumulated across generations. It is the mechanism you are proposing, occurring spontaneously and adversarially. Verify both primary sources directly before citing — secondary coverage of this incident varies in its numbers.

Greshake, K. et al. Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. AISec 2023.
For the memory-poisoning threat model in section 29.

Carlini, N. et al. Extracting Training Data from Large Language Models. USENIX Security 2021.
For the private-information-in-persistent-memory concern.
