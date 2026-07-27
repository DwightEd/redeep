# ReDeEP-Token on fixed RAGTruth responses

This overlay keeps the released ReDeEP source tree unchanged and adds a
verified, resumable evaluator for modern checkpoints. It separates the frozen
released artifact from the explicitly labeled checkpoint-transfer adaptation
used for newer models.

## What is preserved

- The official ReDeEP system/user chat prefix.
- Teacher-forced scoring of fixed responses.
- The 32 released Meta-Llama-3-8B-Instruct Copying-Head candidates.
- Top-10% prefix-token selection for each candidate head.
- ECS as the cosine similarity between the mean final-layer representation of
  attended prefix tokens and the current causal state.
- PKS from LogitLens distributions before and after every FFN.
- The released non-standard divergence direction
  `KL(M || P) + KL(M || Q)`, vocabulary mean, multiplied by `1e6`.
- Min-Max normalization and the score `PKS - beta * ECS`.
- The released 12,000-character prompt slice, FP16 execution, and eager
  attention path.

## What necessarily changes

The ICLR paper reports response AUROC after averaging token scores. It does not
report RAGTruth span-token AUROC or per-task token AUROC. This implementation
assigns each causal state to the next fixed response token, maps half-open
character spans to overlapping subword tokens, and reports QA, Summary,
Data2txt, task-macro, support-weighted, and pooled token AUROC.

The default `train-transfer` protocol is used for Llama-3.1 and Qwen3. The
released Meta-Llama-3-8B candidate set is transferred and re-ranked only on
the complete 2,515-response RAGTruth train split. This is a cross-checkpoint transfer for
Llama-3.1 and both a cross-checkpoint and cross-architecture transfer for
Qwen3. The released values of three heads, thirty FFN layers, and `beta=0.4`
are benchmark-informed and fixed before the target test responses are
evaluated. Min-Max ranges are fitted on train and then frozen. No label from
the target Llama-2 test responses is used for configuration.

The separate `released` mode applies the exact Meta-Llama-3-8B artifact,
including its selected components and Min-Max ranges. It is a
frozen-configuration verification mode, not the primary result for
Llama-3.1. The original regression script selects components and ranges on
test and contains a dataframe slicing error; neither behavior is carried into
the target-test-label-free transfer comparison.

The upstream code computes ECS over the entire chat prefix rather than a
separate retrieved-evidence mask. This implementation retains that behavior.
It evaluates all 450 Llama-2-7B-Chat test responses by default. Correct
half-open span alignment replaces the released script's off-by-one label
mapping and is the only change to token supervision.

The upstream repository does not provide Qwen3 Copying-Head discovery or an
identification script for arbitrary new checkpoints. Both target models
therefore require the explicit `--allow-checkpoint-transfer` flag. Qwen3
additionally requires `--allow-cross-architecture-transfer`. Neither
target-model result should be described as an official checkpoint
reproduction.

## Remote run

The script reuses the environment created for the LUMINA experiment and never
updates or cleans the Git repository.

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/ReDEeP-ICLR
bash scripts/run_llama31_on_llama2_ragtruth.sh smoke
nohup bash scripts/run_llama31_on_llama2_ragtruth.sh full \
  > /share/home/tm902089733300000/a903202310/lys/results/redeep_token/redeep.nohup.log \
  2>&1 &
```

For the explicitly labeled Qwen3 adaptation:

```bash
bash scripts/run_qwen3_on_llama2_ragtruth.sh smoke
nohup bash scripts/run_qwen3_on_llama2_ragtruth.sh full \
  > /share/home/tm902089733300000/a903202310/lys/results/redeep_token/qwen3.nohup.log \
  2>&1 &
```

Rerunning `full` resumes from complete per-response gzip feature files. It does
not run `git pull`, reject a dirty tree, or recompute completed responses.
The resume fingerprint binds the adapter source, installed model
implementation, RAGTruth files, tokenizer artifacts, and every model-weight
shard, so incompatible features cannot be silently reused.

The concise result table is written to:

```text
/share/home/tm902089733300000/a903202310/lys/results/redeep_token/llama31_on_llama2/final_report.md
```

The complete protocol, calibration, metrics, and token predictions are stored
beside it in `protocol.json`, `calibration.json`, `results.json`, and
`predictions.jsonl.gz`.

## Fidelity checks

The released Llama-3 candidate-head and hyperparameter JSON files are checked
against fixed SHA-256 values. The integration oracle runs the vendored
Transformers 4.42 forward with `knowledge_layers` and
`output_attentions=True`, then compares its tokenwise ECS and PKS with the
memory-bounded hook extractor under Transformers 4.51.3 using the same tiny
Llama weights. Formula, alignment, data-cohort, and configuration contracts
are covered by the regular test suite.
