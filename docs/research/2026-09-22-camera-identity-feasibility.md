# Camera Identity Feasibility Record (issue #89)

Date: 2026-09-22. Machine: operator's Mac (fixed host).
Status: path (c) CONVERGED per commander ruling `d-20260922083250331816-8`
(assertion-only; no auto-select; `--device local` stays unauthorized).

## Revision trail (replaced, not silently relaxed)

- Original gate: 3-item all-or-infeasible (set-equivalence both states /
  sort-equivalence / cross-reboot stability) — miss any one => (b).
- Replaced by: (1) set-equivalence both states (present PASS, absent
  pending operator); (2) sort-equivalence PASS; (3) identity-stability
  CONDITIONAL (design-question check: no built-in flag exists, so
  pinning is practically unavoidable; mismatch-fails-loud is the
  guarantee; cross-reboot becomes unverified-limitation, not blocker);
  (4) NEW virtual-camera-uncovered as known limitation.
- Path (a) pyobjc explicitly NOT approved; path (b) remains the fallback
  if any revised item fails.

## Evidence (measured, not assumed)

1. **OpenCV cannot enumerate names/uniqueIDs.** cv2 5.0.0 measured:
   `CAP_PROP_GUID` and `CAP_PROP_HW_DEVICE` return `-1.0` on opened
   devices; `videoio_registry` lists backends only, no device surface.
2. **OpenCV index semantics (source-level).**
   `cap_avfoundation_mac.mm:362-383`: `devicesWithMediaType:Video`
   UNION `devicesWithMediaType:Muxed`, sorted by uniqueID string
   (`[d1.uniqueID compare:d2.uniqueID]`), then `devices[cameraNum]`.
   Verified against 5.x AND master (commander cross-checked :379-384).
   On this machine `D9B9…` (iPhone) < `EAB7…` (built-in), so iPhone
   presence deterministically reorders indices — drift is NOT uncertainty.
   This single rule explains all four historical observations (09-17 /
   09-21 / 09-22 / today).
3. **macOS layer identity, three-way consistent.** `system_profiler`
   `spcamera_unique-id` == `AVCaptureDevice uniqueID()` byte-identical
   (built-in `EAB7A68F-…`, iPhone `D9B9EBF1-…`); enumeration opens no
   camera. Re-runnable via `scripts/verify_camera_identity.py`.
4. **Sort equivalence.** Measured uniqueIDs all-ASCII; Python `sorted()`
   == live AVCaptureDevice array order (2026-09-22).
5. **Design question.** No built-in boolean exists anywhere:
   `system_profiler` has only 3 fields; `AVCaptureDevice position()`
   returns 0 (Unspecified) for both cameras. `spcamera_model-id` equals
   the localized name on this machine — no independent stable key.

## A2 shape constancy (measured 2026-09-23, operator live-released)

- iPhone present: predicted built-in index 1; 5 independent opens all
  `720x1280`.
- iPhone powered off (operator-confirmed index 0 is built-in): 5
  independent opens all `720x1280`.
- Merged: **10/10 constant cross-state**. No frames saved, no pixels
  viewed; zero failures. Shape check is therefore mandatory on
  true-device runs (same level as the uid pin) — see contract item 4.

## Known limitations (accurate, not silent)

- Cross-reboot uniqueID stability: UNVERIFIED (no reboot performed).
  Guard: mismatch refuses LOUD.
- Virtual cameras (OBS etc.): UNCOVERED (none on this machine).
  Guard: set-count comparison refuses on surprise devices.
- iPhone powered off: `system_profiler` STALE — still lists 2 devices
  (built-in + powered-off iPhone) while AVFoundation Video ∪ Muxed
  drops to 1. The count check fail-closes (2 != 1, exit 2), so a
  legitimate built-in run is refused until the profiler refreshes.
  Availability limitation, recorded not fixed; never describe the
  refusal as an identity-assertion success.
- `Muxed` set membership: any future enumeration source must cover
  Video UNION Muxed, or the predicted index silently miscomputes.
