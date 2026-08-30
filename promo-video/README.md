# FIT-GGUF Chinese launch film

Deterministic 62-second Remotion film for the first FIT-GGUF release. The
master composition is 1920x1080 at 30 fps and uses only local vector/chart
assets; no model inference or GPU rendering is required.

## Structure

| Time | Beat |
| ---: | --- |
| 00:00-00:07 | Brand reveal |
| 00:07-00:14 | Preset-size problem |
| 00:14-00:21 | Continuous size slider |
| 00:21-00:28 | Analyze / plan / quantize workflow |
| 00:28-00:35 | Fourteen release tiers |
| 00:35-00:42 | Post-oracle byte prediction |
| 00:42-00:49 | Measured FIT-12G quality point |
| 00:49-00:56 | Honest scope and limitations |
| 00:56-01:02 | Closing lockup |

## Commands

```bash
npm install
npm run check
npm run studio
npm run render:preview
```

`render:preview` deliberately renders the full 1920x1080 composition with a
single worker. Passing a smaller width/height changes the CSS viewport and is
not a valid layout preview.

## Current audio status

The checked visual preview contains a near-silent AAC placeholder track. Add
licensed music, purpose-built sound design or a final Mandarin voice-over only
after the visual cut and factual wording are approved.

