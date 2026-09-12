# Nex-N2.5-mini-abliterix-v8-t17 — conversion provenance and the MTP exception

## Sources

| Artefact | Origin | Note |
|---|---|---|
| Text BF16 GGUF | `Nex-N2.5-mini-abliterix-v8-t17` (abliterix direct+EGA bake) | 40 blocks, 733 tensors |
| mmproj BF16 GGUF | `Nex-N2.5-mini` (original) | vision tower is untouched by the abliteration, so the projector is exactly the source's |

The abliteration export is **text-only** (`text_only = true`): it carries the text
decoder and `lm_head` but not the 333 `model.visual.*` tensors. Per D-0025 the
vision capability is restored by shipping the projector built from the original
weights, which the abliteration never modified.

## MTP / NextN exception (owner-approved, 2026-09-11)

The upstream `config.json` declares `text_config.mtp_num_hidden_layers = 1`, but
the checkpoint ships **no `mtp.*` tensors** — 40 blocks, nothing at index 40. The
same holds for `orcarouter/Qwen3.8-27B-Uncensored`, whose released GGUF carries
`block_count = 64` and no `nextn_predict_layers` field.

`convert_hf_to_gguf.py` trusts the declaration, so a naive conversion writes
`block_count = 41` + `nextn_predict_layers = 1` and the runtime then fails with

    check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found

Owner decision: **convert by the actual tensor count and record the discrepancy**
(metadata must describe the file). This conversion therefore runs with
`--no-mtp`, which here means exactly one thing: do not claim a 41st block that
does not exist in the source. No capability is removed because none was present
to keep. Should upstream ship `mtp.*` weights, this exception is void and the
conversion must carry them (D-0025).
