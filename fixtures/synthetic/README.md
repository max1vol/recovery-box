# Synthetic pose integration fixtures

`squat-standing.jpg` and `squat-down.jpg` are generated, fictional,
non-patient images for the explicit laptop pose-client integration launcher.
They contain no captured camera data and may be inspected or copied during a
hardware integration run.

The ordinary test suite must not open these files through MediaPipe, construct
a native pose runtime, or access camera/audio hardware. Unit tests should inject
numeric fakes at the existing vision boundaries instead.

The images were generated for RecoveryBox on 2026-08-25 and normalized to
640 by 480. The down fixture uses an intentionally exaggerated, front-facing
horse stance so both knee angles are visible to the deterministic 2D tracker;
it is not a prescribed exercise pose. They are intentionally not
representative clinical data and must not be used to evaluate pose-model
accuracy or rehabilitation safety.

Reviewed fixture SHA-256 values:

- `squat-standing.jpg`:
  `ad0e809c1a181927b10ec946a9b6faf8498015d96988b0867a5bf01b365ffd91`
- `squat-down.jpg`:
  `b8d1885ba8f78256c8dabcff87bb2d55f5ffd073d2ab12420f8da191b127cd95`
