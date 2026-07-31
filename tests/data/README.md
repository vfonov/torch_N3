# Test volumes

The images the suite runs on, checked in so that it needs nothing installed to
read them.

They are **byte-for-byte the same images** as the ones N3 ships — converted
from gzipped MINC1 to MINC2, which is the only format `minc2_simple` opens.
Doing that conversion on every test run meant every test needed `mincconvert`
on `PATH`; doing it once means none of them do.

| File | Came from |
|---|---|
| `chunk.mnc`, `chunk_mask.mnc` | `legacy/N3/testing/` — the small 91×52×50 volume, for fast per-block tests |
| `brain.mnc` | `legacy/N3/testing/` — the input for the reference regression |
| `brain_nu_ref.mnc` | `legacy/N3/testing/` — `nu_correct`'s own output, the regression target |
| `icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc` | `/opt/minc/*/share/N3/` — the average brain mask `-auto_mask` reaches for |

`brain_nu_ref_legacy.mnc` is the exception: it is not one of N3's files but one
of ours, written by `tests/regenerate_reference.py`. It is what this pipeline
produces on `brain.mnc` with the original C++ blocks driving it, under the
fixed protocol in `tests/inputs.py`, and it exists so that another machine,
another BLAS or a GPU can be held to it — see `tests/test_reproducibility.py`.

It is the one volume here stored `float64` rather than 16-bit, and the only
reason for its 6.7 MB. Everything else in this directory is a record of what
some N3 program *wrote*, at the precision it wrote it; this is a record of what
the pipeline *computed*. Rounding to 16 bits would put in an error orders of
magnitude above the difference the test measures, so it would have meant
comparing against the rounding.

Regenerating it rewrites MINC's `ident` attribute (user, host, timestamp, pid),
so `git diff` reports the file as changed even when nothing about the image
did. The voxel data is reproducible; those few header bytes are not. Compare
the data, or `git checkout` the file, rather than committing a header churn.

To reproduce, with the MINC toolkit on `PATH`:

```bash
for f in chunk chunk_mask brain brain_nu_ref; do
    mincconvert -2 -compress 9 -clobber legacy/N3/testing/$f.mnc.gz tests/data/$f.mnc
done
mincconvert -2 -compress 9 -clobber \
    /opt/minc/1.9.18.13/share/N3/icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc.gz \
    tests/data/icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.mnc
```

The conversion is lossless — voxel values, `start`, `step` and direction
cosines all compare equal — so the answers recorded in `tests/reference/`
are unaffected by it, and `python3 -m tests.regenerate_reference` reproduces
them from these files unchanged.

`legacy/N3/testing/block.mnc.gz` is deliberately *not* converted: reading
MINC1 and gzip on the fly is a feature of `torch_n3.volume.load_volume`, and
`test_gzipped_minc1_input_is_readable` needs a file in that format to exercise
it. That one test skips without `mincconvert`.
