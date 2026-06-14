# Canonical Patch Protocol

The canonical protocol prevents repair rates from being distorted by repeated retries, duplicate
patches, and no-patch reruns.

For each bug and each model-agent pair:

1. Keep one formal repair unit.
2. If the main run generates an applicable patch, select it as the canonical patch.
3. If the main run produces no patch, a retry may fill the same repair unit.
4. Duplicate patches are not counted as additional attempts.
5. No-patch outputs count as repair failures but do not require build validation.
6. Generated and applicable canonical patches enter executable validation.
