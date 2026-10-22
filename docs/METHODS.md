# Method and allocation ablation

MA-RAG-int samples eight responses per round. If responses disagree, its query
agent produces up to four retrieval queries, MedCorp documents are reranked,
and the following round considers the question, evidence and previous answers.
The harness stops at consensus or eight rounds.

EDR evaluates constrained one-token option distributions before rounds 2–8:
`P_E` from question/options/evidence, `P_H` from question/options/history, and
`P_EH` from all three inputs. Define `TV(P,Q)=sum_a |P(a)-Q(a)|/2` and
`s=max(0,TV(P_EH,P_E)-TV(P_EH,P_H))`. If the unique highest-probability option
under `P_E` differs from that under `P_EH`, route
`m=min(6,floor(6s+0.5))` evidence-only responses and `8-m` history-informed
discussion responses. Ties, unchanged top options and probe failures yield
`m=0`. Both channels retain eight solver responses in total; no weights are trained.

The paired Qwen MedXpertQA evaluation shares all 490 initial eight-answer
outputs and entropy values. Successful first-query and retrieval stages
preserve baseline order, followed by the same descending-entropy history.
The initial query failure on question 29 remains a failure and counts
incorrect in the 490-question result. Run-specific material is recorded in
each benchmark directory.

The gated-random allocation ablation uses the same E/H/EH probes and
positive-route definition on its own trajectory. If the reference adaptive
count is positive, it selects `m_random` from 1–6; otherwise it uses zero
independent and eight discussion responses. Allocation seed 42 and the
realized counts are retained in the trace. Model, prompts, retrieval policy,
voting, stopping and per-round response count are shared between strategies.
After their common initial state, each strategy computes its own probes and
generates its subsequent responses. Complete-run endpoints are compared on
all 490 question IDs.

MedLA uses one shared ordered evidence block and one shared initial agent
pool per question. EDR maps the eight-response reference allocation to the
native representative count, while the host retains its selection, discussion
and final-decision rules. See [MedLA integration](MEDLA_INTEGRATION.md).
