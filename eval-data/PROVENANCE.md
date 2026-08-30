# Evaluation Slice Provenance

The five fixed 64 KiB evaluation slices (65,536 UTF-8 characters each) used
by every KL measurement in this repository, preregistered as the M9 slice
set and reused unchanged across M9-M16 and P1-P6. Each slice is cut from a
documented source at a documented character offset; the M11/M13 holdout
sets are cut from the same sources at disjoint offsets by
`scripts/make_holdout_slices.py` (SHA-256 records in the experiment dirs).

| Domain | File | Source | Character offset |
| --- | --- | --- | ---: |
| wiki_test | `kl-eval-64k.txt` | wikitext-2-raw `wiki.test.raw` | 0 |
| wiki_valid | `kl-eval-valid-64k.txt` | wikitext-2-raw `wiki.valid.raw` | 0 |
| chinese | `kl-eval-cn-64k.txt` | calibration corpus `combined_cn_medium.parquet` | 3,173,914 |
| code | `kl-eval-code-64k.txt` | calibration corpus `code_medium.parquet` | 12,536,883 |
| agent_chat | `kl-eval-agent-64k.txt` | calibration corpus `agentworld_clean_quick.txt` | 998,087 |

## SHA-256

```
d400318e1ad6981e3fa332514ae5a59e98d888592428d152a0ffc3ceb135620e  kl-eval-64k.txt
9b455800b98525e0f6ec3ac18f4a6b789622f7d9ae381e5a174f5c3d42173402  kl-eval-valid-64k.txt
a7584dc67f2e3050d42326da91f2801bf29c381596eb1364896490c98bad56d5  kl-eval-cn-64k.txt
da9cae0047be52338c7710d7b6cc00354f05d2c8009b9bb3d7914f08d65a4084  kl-eval-code-64k.txt
01c79b525330a642d89d0b8a00f0b42931ce0b94546c894047d8da041c86188b  kl-eval-agent-64k.txt
```

## License notes

- The wiki slices are excerpts of WikiText-2 (Salesforce wikitext,
  CC-BY-SA-SH 3.0); the excerpts are used here solely as evaluation text
  with attribution to the WikiText authors.
- The chinese / code / agent_chat slices are short evaluation excerpts from
  the project's imatrix calibration corpora. Their provenance and
  construction are documented in the calibration-data repository this
  project was developed against; the excerpts are provided for exact
  reproduction of the reported KL numbers.

## Protocol

All measurements: `llama-perplexity -ngl 99 -t 16 -c 512 -b 512
--kl-divergence --kl-divergence-base` with the pinned llama.cpp runtime,
against BF16 logits of the same source model on aligned input. See
`DECISIONS.md` (M9) and the P4 README for the frozen protocol text.
