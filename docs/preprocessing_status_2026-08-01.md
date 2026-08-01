# OurDataset FaceXFormer preprocessing status

Audit time: 2026-08-01 18:46-19:00 CST  
Server: `bbdbfbdr4`  
Output: `/mnt/adata/OurDataset_RhythmFormer_FaceX_128`

## Final status

The attempt-03 pipeline and postflight validation completed successfully.

| Item | Result |
| --- | ---: |
| Mapping rows | 618 |
| Validated output directories | 607 |
| Skipped sessions | 11 |
| Failed sessions | 0 |
| Output size | 125 GB |
| `postflight_v3.exit` | 0 |
| Completion marker | 2026-07-31 21:59:15 CST |

The resume drill re-read all 607 successful outputs without changing them:
`607 resumed`, `11 skipped`, `0 failed`. Each successful directory has exactly
one `vid.avi`, one `ground_truth.txt`, and one `.facex_quality.json`.

The audited sample `subject491` is FFV1, 128x128 BGRA, 50 FPS, and contains
18,661 frames. All 607 generated videos use the same per-session output
contract; FPS follows the source recording.

## Skipped sessions

Nine sessions lack a required alignment CSV:

| Session | Original subject | Activity | Reason |
| --- | ---: | --- | --- |
| subject113 | 203 | bike_level1 | alignment CSV missing |
| subject114 | 203 | bike_level3 | alignment CSV missing |
| subject115 | 203 | bike_level5 | alignment CSV missing |
| subject116 | 203 | rotate | alignment CSV missing |
| subject117 | 203 | speak | alignment CSV missing |
| subject118 | 203 | static_level1 | alignment CSV missing |
| subject119 | 203 | static_level3 | alignment CSV missing |
| subject120 | 203 | static_level5 | alignment CSV missing |
| subject284 | 473 | bike_level3 | alignment CSV missing |

Two sessions were explicit operator exclusions:

| Session | Original subject | Activity | Reason |
| --- | ---: | --- | --- |
| subject381 | 656 | bike_level5 | detection rate 0.5995, below the 0.80 gate |
| subject402 | 703 | static_level5 | corrupt H.264 source; only 7,552/7,560 source frames decodable |

These are recorded as `skipped`, not preprocessing failures.

## Demonstration subject 798

The mapping was resolved by `original_subject = 798` in
`session_mapping.csv`, not inferred from directory ordering.

| Session | Activity | Frames | Detection | Interpolated frames |
| --- | --- | ---: | ---: | ---: |
| subject491 | bike_level1 | 18,661 | 100% | 0 |
| subject492 | bike_level3 | 18,599 | 100% | 0 |
| subject493 | bike_level5 | 18,370 | 100% | 0 |
| subject494 | rotate | 6,343 | 99.795% | 13 |
| subject495 | speak | 6,398 | 100% | 0 |
| subject496 | static_level1 | 6,331 | 100% | 0 |
| subject497 | static_level3 | 6,352 | 100% | 0 |
| subject498 | static_level5 | 6,612 | 100% | 0 |

All eight sessions passed integrity and temporal RGB correlation checks.

## Manual visual review

Attempt 03 recovered ten alignment sessions and generated contact sheets for:

`subject033`, `subject097`, `subject121`, `subject122`, `subject123`,
`subject124`, `subject125`, `subject126`, `subject127`, and `subject146`.

Their numerical quality gates passed, but
`visual_review_v3_recovered/candidates.csv` still marks them
`pending_manual_review`. This is the only remaining human review item; it does
not change the successful postflight result.

## Git tracking audit

- `/mnt/adata` and the 125 GB output directory are outside the source
  repositories and are not tracked by Git.
- The core preprocessing implementation was tracked in
  `remote-physiological-signal-data-preprocessing.git`, branch
  `feature/facexformer-preprocess`, at commit
  `fa117605e302da947f11cbebec5f594f560f402d`.
- Attempt-03 operations and postflight scripts were stored beside the generated
  output and were not part of that source repository before this archive.
- A copied worktree at
  `/mnt/adata/facexformer_preprocess_worktrees/RhythmFormer` has a broken `.git`
  pointer to a former macOS path and cannot report Git status on the server.
- FaceXFormer is an external dependency at commit
  `10fe8291f8a64e2ca1daf938e3e0007bd860303b`; its checkpoint and runtime cache
  are intentionally excluded.

No preprocessing process was active at audit time. The visible Python GPU
processes were PhysFormer training jobs, not this preprocessing pipeline.
