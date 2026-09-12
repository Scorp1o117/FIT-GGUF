# Nex-N2.5-mini-abliterated: fourth-model onboarding — calibration + four fidelity tiers

**Date**: 2026-09-11/12 · **Runtime**: llama.cpp b10666 (`4e97ac86e`, ROCm) · **Status**: in progress

The first onboarding of a **Qwen3.5-MoE hybrid** source (40 blocks, 30 linear-attention +
10 full-attention, 256 fused experts, 3B active) and the first where the abliterated
weights come from this project's own companion pipeline (`abliterix` direct + EGA).

## Source and provenance

| Item | Value |
|---|---|
| Abliterated weights | `Nex-N2.5-mini-abliterated` (direct + EGA, 160-token convention; shipped 12/100 refusals @ KL 0.0662) |
| Text BF16 GGUF | `artifacts/source/Nex-N2.5-mini-abliterated-BF16.gguf` — 40 blocks, 733 tensors, sha256 in `state/source-sha256.txt` |
| Vision projector | `artifacts/source/Nex-N2.5-mini-abliterated-mmproj-BF16.gguf` (902 MB) — built from the original checkpoint |
| imatrix corpus | `APEX-imatrix-Small.txt`, 500 chunks (contract default) |
| Eval data | `eval-data/` five frozen slices (wiki_test, wiki_valid, chinese, code, agent_chat) |

### Two capability notes (D-0025)

1. **Vision tower restored.** The abliterix export is text-only, so the 333
   `model.visual.*` tensors are absent from the text GGUF. They are untouched by the
   abliteration, so the projector built from the original checkpoint is exactly the
   source's projector. It ships beside the tiers; FIT's tier search governs the text
   model only.
2. **MTP exception (owner-approved).** The upstream config declares
   `text_config.mtp_num_hidden_layers = 1` while shipping no `mtp.*` tensors — 40
   blocks, nothing at index 40 (the same inconsistency exists in
   `orcarouter/Qwen3.8-27B-Uncensored`, whose release carries `block_count = 64` and no
   `nextn_predict_layers`). Converting per the declaration produces `block_count = 41`
   and the runtime then fails with `check_tensor_dims: tensor 'blk.40.attn_norm.weight'
   not found`. Owner decision: convert by the actual tensor count and record the
   discrepancy, so the metadata describes the file. No capability is removed because
   none was present. See `state/conversion-notes.md`; if upstream ships `mtp.*`
   weights this exception is void.

## Method

1. Conversion with the repository's pinned `convert_hf_to_gguf.py` (note 2 above); a
   one-chunk `llama-imatrix` smoke test confirms the artifact loads before any
   calibration spend.
2. `fit calibrate` — five aligned BF16 references, a standard preset ladder with at
   most four gap probes per tier, floors derived inside the tier windows
   `W(K) = [0.85K, 1.15K]`.
3. `fit fidelity-search` per tier, run from the calibration bundle's bracket evidence
   (`state-artifact-manifest.txt`), one tier at a time.
4. Each tier re-quantized from its recorded recipe and re-evaluated on its own bytes.

Execution profile: `--n-gpu-layers 30 --threads 16` for every step in this session.
Scratch and subprocess logs live on ext4 (`/home/s117/fit-scratch`, `/home/s117/fit-logs`);
`/run/media/s117/OS` is ntfs3 and the project documents a kernel BUG when llama.cpp's
unbuffered stderr is redirected onto it.

## Result

All sizes are the delivered bytes; every tier's KL is the 3-token
teacher-forced macro KL against the five BF16 references, measured on the
artifact's own bytes.

| Tier | Size | Bytes | Recipe / primary type | Macro KL | Same-top | Gate (KL ≤) | Evals |
|---|---|---|---|---|---|---|---|
| QUALITY | 23.12 GiB | 24,823,238,560 | FIT recipe, `Q5_K_M` (base preset; dominant type Q5_K is a bare tensor type) | **0.0465** | 93.36% | 0.05 | 3/8 |
| BALANCED | 15.57 GiB | 16,714,051,488 | FIT recipe, `IQ3_S` | **0.0998** | 89.32% | 0.10 | 5/8 |
| COMPACT | 13.76 GiB | 14,772,088,736 | FIT recipe, `IQ2_S` | **0.1495** | 85.81% | 0.15 | 8/8 |
| MINI | 12.69 GiB | 13,623,758,752 | `IQ3_XXS` (ladder lower bound; zero-override native preset) | **0.1860** | 83.92% | 0.20 | 2/16 |

All four delivered tiers carry **G2 delta +0**: the delivered bytes equal the
re-finalized prediction exactly. All four are **KL-bound** (`active constraint: kl`), so the tier's quality
limit decided the size rather than a size floor.

MINI's answered size is exactly the `IQ3_XXS` preset (13.62 GB), i.e. a
zero-override native preset whose recipe the search verified rather than
re-derived; its two evals confirm the bracket rather than searching it.

The calibration curve for this model, which is what the sizes fall out of:

| point | size | macro KL | | point | size | macro KL |
|---|---|---|---|---|---|---|
| IQ2_XXS | 8.85 GiB | 0.4794 | | IQ4_XS | 17.44 GiB | 0.0833 |
| IQ2_XS | 9.79 GiB | 0.3745 | | Q4_K_M | 19.71 GiB | 0.0796 |
| IQ2_M | 10.86 GiB | 0.2850 | | Q5_K_M | 23.03 GiB | 0.0586 |
| IQ3_XXS | 12.69 GiB | 0.1860 | | Q6_K | 26.56 GiB | 0.0384 |
| IQ3_XS | 13.49 GiB | 0.1594 | | Q8_0 | 34.37 GiB | 0.0334 |
| IQ3_M | 14.38 GiB | 0.1498 | | | | |

### MINI: first attempt budget_exhausted, retried

The first MINI search used the ladder `IQ2_XXS,Q4_K_M`, whose lower anchor fails
badly (0.4794 at 8.85 GiB). The walk therefore had to climb the whole distance to
the first PASS and ran out of budget one step after finding it:

    budget_exhausted | best observed PASS 13.56 GiB (kld 0.1957) | 8/8 evals
    NOT auto-delivered (artifact withheld; adjudication required)

The retry (`fs/mini2/`, `run-mini-retry.sh`) uses `IQ3_XXS,Q4_K_M`, whose lower
anchor already passes (12.69 GiB @ 0.1860), so the search confirms that bracket
instead of walking up — with `--profile precise` for headroom. It returned
`verified_pass` in **2 evals**:

    verified_pass | tier mini | minimum verified PASS 12.69G (kld 0.1860, top 83.92%) | 2/16 evals

**The lesson generalises**: a tier whose ladder lower anchor already passes the
gate is answered in a couple of evals; a ladder anchored far below the crossing
spends its whole budget climbing (stride is `(max_size - min_size) // 16`, so a
narrow ladder makes the climb slower, not faster).

## Published artefacts

The delivered GGUFs live in the models tree, not in this repository (the repo
carries the records; `.gitignore` excludes `*.gguf`):

    /run/media/s117/OS/Models/Nex-N2.5-mini-abliterated-FIT-GGUF/

| File | SHA-256 (first 20) |
|---|---|
| `…-FIT-QUALITY-23.12GiB-Q5_K_M.gguf` | `2450389c5bb4056cdf1b` |
| `…-FIT-BALANCED-15.57GiB-IQ3_S.gguf` | `6436ee9c8c0af3defabb` |
| `…-FIT-COMPACT-13.76GiB-IQ2_S.gguf` | `e32c3b16269896a93741` |
| `…-FIT-MINI-12.69GiB-IQ3_XXS.gguf` | `a96bb1004ff085f2b0c4` |
| `calibration/calibration-imatrix.gguf` | `763de6b733e3293b4a9d` |

Full list in `state/tier-artifacts-sha256.txt`; the release folder carries its own
`SHA256SUMS` and a README with usage. All four hashes were re-verified after the
move and match the recorded values.

The **vision projector** (`Nex-N2.5-mini-abliterated-mmproj-BF16.gguf`, 0.84 GiB) sits
beside the tiers; FIT's search governs the text model only. The converter input
(`Nex-N2.5-mini-abliterated-BF16.gguf`) stays in `artifacts/source/` as the
pinned source of record.

## Reproduction

`run-four-tiers.sh` in this directory is the exact driver used, including the
free-space gate that keeps a 12-40 GB candidate from filling the ext4 volume.
