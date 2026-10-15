# Changes

Notable changes to `pandorafits`, newest version first. Every change that affects users, file contents or the API gets a line under the version in development.

## v0.10.0 (in development)

### Changed Level 2 values (regenerate Level 2 files)

- `TIME` is the middle of each exposure for NIRDA (1.9 s earlier than before) and VISDA (half a co-add later), marked by `TIMEPIXR = 0.5` in the `TIME` header; the pointing in `VECTORS` is averaged over each exposure. Regenerate Level 1 files too; files without `TIMEPIXR` use the old convention.
- NIRDA `BACKGROUND` leaves out the subarray edge columns and clips sources per row; it was 0.6 to 1.0 ct/s/pixel too high.
- `APERTURE`, `MODEL_IMAGE` and `CONTAM`/`TOTAP`/`COMPLTE` scale the scene stars by each detector's own band and zero point (2MASS J for NIRDA, Gaia BP for VISDA, from pandoraref's `MAG_SRC`). Stars without that magnitude get zero flux, counted in the new `APERTURE` cards `MAGSRC` and `NOMAG`.
- The NIRDA `aperture` property keeps every row and excludes only QUALITY-flagged pixels (affects `APCOMP1`, `TARGCOMP`, `LC_MED`).
- Truncated NIRDA files keep every complete integration; their `FRMSTOT` is a frame count.
- The stored WCS is the one used for `POSX`/`POSY`, `VECTORS` and `APERTURE`.
- RA and roll are averaged on the circle (Level 0 database, `VECTORS`, WCS center); `target_sep` is the angular separation on the sky (changes `ONTARG`, `TARGTIME`).
- SOC report scores: `TARGCENT` replaced by `TCENT10`/`TCENT20`/`TCENT30`; `TARGCOMP` compares fractions; missing catalog values are written as -99 (magnitudes) or -1 (Teff, log g).

### Fixed

- Database crawls on Windows: Level 0 entries stored the full path as the file name and an empty folder, so re-crawls failed and duplicate downloads were not skipped. Rebuild Level 0 databases made on Windows.
- Files in `engineering_data` or `staging` folders are skipped by the crawls, as intended; banned folder names are now a separate list, `BANFOLDERS`, matched only against folders below `DATA_DIR`.
- Engineering and payload telemetry databases ingest every day (only one day was kept), store UTC times (packet times are TAI; they were stored 37 s late), match columns by name, update days that are re-ingested, and work on Windows. Rebuild these databases.
- Level E no longer reprocesses recent files on every run (or files with bad checksums); a remade engineering file replaces its entry.
- Chunked processing (`nchunks`, `chunk`) assigns each file to a chunk by a stable hash of its name, so chunks no longer skip or repeat files as other chunks finish.
- `writeto` writes fresh checksums by default and sets `NEXTEND`; the Level 2 `ASTROMETRY` header no longer repeats `EXTNAME` or carries stale checksums.
- A calibration step (bias, dark, flat, ...) is not applied twice.
- `SNRInspector` no longer crashes on an empty target aperture; an empty aperture is logged.
- `get_bkg` used `np.median` on masked arrays, which ignores the mask.
- `_append_wavelength` without `VECTORS`, and `panels_to_cube` for borderless panels.

### Added

- Demos: `demos/01_visda_reduction.ipynb`, `02_nirda_reduction.ipynb`, `03_visit_analysis.ipynb`.
- `utils.wrap_deg`, `circular_mean_deg`, `circular_median_deg`, `CircularMeanAggregate`; `processing.get_scene_flux`, `DetectorSkyScene`; `PandoraHDUList.exposure_windows`; `nirda.get_nirda_read_mid_times`; `utils.split_path`, `unique_by_name`; `database.BANFOLDERS`, `is_banned`; `database.mixins.ProcessingQueueMixins`, `file_chunk`.
- Tests for the above; the dummy-file tests write to a temporary folder.
