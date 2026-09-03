# Archive: R2 compact grid at the ORIGINAL B_FS (12,206,255,328 = 11.37 GiB)

The 2026-09-03 evening R2 sweep ran the compact tier at the P1 answer
B_FS = 12,206,255,328 (11.37 GiB). Its letter verdict was FAIL: two healthy
grid PASSes landed below the tolerance line B_FS - 128 MiB:

- R2A-compact-off-192: delivered 11,991,706,848 (11.17 GiB), kld 0.1486, top 89.09 -> PASS
- R2A-compact-off-128: delivered 12,023,164,128 (11.20 GiB), kld 0.1493, top 89.10 -> PASS

The crossing region shows eval-noise interleave with the tier margin
(four observations within +-0.005 of the 0.15 anchor across ~105 MB:
PASS 0.1486 / PASS 0.1493 / FAIL 0.1502 @ 12,038,647,008 (P1 FS02) /
FAIL 0.1582 @ 12,076,248,288 / FAIL 0.1539 @ 12,072,070,368 (P1 FS03)).

Per the recovery plan these grid points are renamed R2A-* (eval logs live in
logs/eval-orcarouter-R2A-compact-*.log; sizes backfilled into
results/artifact-manifest.txt with marker `eval-only` because the tmpfs
artifacts were released after evaluation and their SHA-256 was not recorded;
each artifact is deterministically reproducible from the *-plan-recipe.json /
*-plan-tensor-types.txt in this directory).

Note: R2A-compact-off-256 was clipped into the MINI window by the sweep's
then-current clip rule (nearest healthy window below the target) and
delivered 11,185,470,688 with mini-region metrics kld 0.1923 / top 86.40
(FAIL). R2A-compact-off+64 was interrupted mid-eval (4/5 domains) and is not
admissible evidence; its plan record shows the same delivered size as B_FS
itself (12,206,255,328).

After this archive the compact tier re-runs its fidelity search with the full
evidence (adoption) and the R2 grid is recomputed at the adopted B_FS.
