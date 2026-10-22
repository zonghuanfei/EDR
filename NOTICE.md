# Third-party dependencies

The original MA-RAG framework and its MedXpertQA question file are maintained
by NJU-RL at https://github.com/NJU-RL/MA-RAG. Consult the upstream licenses
and data terms before distribution or rerunning. MedCorp passage text is not
included. The package contains generated experimental trajectories, minimal
retrieval metadata, and the locally implemented EDR/ablation hooks.

Original contributions by Huanfei Zong are licensed as specified in
[LICENSE.md](LICENSE.md): MIT for software, and CC BY 4.0 for experimental
data and documentation. These grants do not relicense third-party code,
prompt text, quoted material, datasets, model weights, or retrieval sources.
Any retained third-party notices and license terms continue to apply.
Dataset-specific provenance is recorded in each benchmark's `DATASET.md`.

The MAD-M2 objective-history-masking comparator transfers the score and
selection mechanism from *Multi-Agent Debate with Memory Masking*
(https://arxiv.org/abs/2603.20215), with reference implementation at
https://github.com/tmlr-group/MAD-MM, commit
`f02069add08280b764d059a2f06ca0043aa093e2`. The added local files are the
MA-RAG and MedLA transfer adapters and experimental records, not a redistribution of
the complete upstream framework or an original claim to its selector.

The MedLA framework-transfer evaluation uses the released MedLA implementation
at https://github.com/alexander2618/MedLA, commit
`5c12cfc8d67170b1f4b131b9c120a54a573c634d`, corresponding to
*MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning
with Large Language Models* (https://doi.org/10.1609/aaai.v40i2.37052).
The package contains the local HTTP/EDR adapter and generated MedDDx-Expert
and MMLU-Med experimental records. It does not bundle the complete upstream framework,
its question bank, model weights or MedCorp source passages. Obtain the
upstream code and dataset separately and observe their original terms.

The complete 1,089-question MMLU-Med test is sourced from the MIRAGE release,
https://github.com/gzxiong/MIRAGE, commit
`392943af99cd94cafd50a0de2e7fca24bbf65494`. Its six medical subjects retain
the upstream order and source identifiers. The package records generated
reasoning, labels, predictions and minimal retrieval metadata; it does not
redistribute the MIRAGE benchmark JSON or the retrieval corpus. Obtain the
question files separately and observe the original MMLU/MIRAGE data terms.
