# Dataset Identity: MedDDx-Expert

- Dataset: MedDDx, Expert difficulty (released file named hard).
- Scope: all 483 unique questions in the MedLA released Expert/hard test file.
- Task: four-option biomedical knowledge and relationship reasoning, including
  gene/protein, pathway and drug relationships with closely related distractors.
- Answer space: single choice A-D.
- Repository: https://github.com/alexander2618/MedLA
- Source commit: `5c12cfc8d67170b1f4b131b9c120a54a573c634d`.
- Source file: `data/medddx/test_hard.jsonl`.
- Source fields: `idx`, `question`, `options`, `answer_idx`, `answer`.
- Public trace `id` and `source_id`: the zero-based source question index, 0-482.
- Scoring gold: `answer_idx`; used after generation for evaluation, not as a
  retrieval query, example selection target or solver input.
- Evaluation: Gemma 3 12B IT; MedLA with shared MedCorp evidence versus the same
  framework plus EDR; completed 2 October 2026.

The directory retains generated experimental data and minimal retrieval
metadata, not the original question bank or MedCorp passages. Obtain original
questions from the pinned upstream repository and follow its data terms.
Package licenses do not relicense upstream data, prompts or model weights.
