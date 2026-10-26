# teval

Evaluation tool for THz time-domain spectroscopy (THz-TDS) measurements, with a GUI that is generated automatically from the
evaluation components.

## Starting the GUI

Install the dependencies (`pip install -r requirements.txt`; numba and threadpoolctl are optional but make the q-space
evaluation much faster) and run from the repository root:

```
python load_with_ui.py eval_scripts/test_script.py
```

Settings are loaded from and saved to `config/<script name>.json`, e.g. `config/test_script.json`, when the window is closed.

Group "Settings file" of the Settings tab:
- "Load settings file" / "Save settings as" switch to another file (e.g. one per sample type). That file is then used for
  saving and on the next start (the GUI and the command line interface), until another file or "Use default settings
  file" is chosen. Settings a loaded file doesn't contain are reset to their defaults.
- "Save settings now" saves to the current file, also when "Save settings on exit" is off.
- "Reset all settings to defaults" resets every component; data and file paths are kept. A single group: right click on
  it and choose "Reset ... to defaults".

"Log level" (Settings tab) sets which messages appear in the message panel; "debug" also shows the diagnostic messages.

## Measurement files

Group "File name format" of the dataset: which files are measurements and what is read from their names. Each file is a
text file with the time axis (ps) in the first and the signal in the second column.

- "Files": glob pattern below the dataset path, default `**/*.txt` (also in sub folders).
- "File name format": how the measurement time, position and other attributes are read from the file name (without
  extension).
  - **Empty (default):** the timestamp at the start of the name and the `<value> mm` parts as the positions, e.g.
    `2025-05-27T23-29-21.791821-20avg_s2_sub-sam-X_25.000 mm-Y_-1.000 mm.txt`.
  - **Template:** the whole name has to match; `{field}` is a field, `*` anything (as little as possible). Fields:
    `{time}` (in the "Time format"; without it the file modification time is used), `{x}` and `{y}` (position in mm, 0 if
    missing), `{name}` (measurement name for the sample groupings by name, instead of the name pattern below) and any
    other name as an extra attribute (a number if it is one; `{field:num}` / `{field:int}` only match numbers).
    The attributes are listed in "Show selection details".
  - **`re:` + regular expression:** the same fields as named groups, searched in the name.
- "Time format": `strptime` format of `{time}`, default `%Y-%m-%dT%H-%M-%S.%f`.
- "Test file name format" logs how the first files are parsed, how many match and examples of the ones that don't.

| File name | File name format | Time format |
|---|---|---|
| `2025-05-27T23-29-21.791821-20avg_s2_sub-sam-X_25.000 mm-Y_-1.000 mm` | empty, or `{time}-{avg:int}avg_s{sample:int}_{layer}-{kind}-X_{x} mm-Y_{y} mm` | default |
| `sam_pow1_25.000_-1.000_20250527_232921_791821` | `{kind}_{power}_{x}_{y}_{time}` | `%Y%m%d_%H%M%S_%f` |
| `scan3_x1.5_y-2.dat` (files `*.dat`) | `{name}_x{x}_y{y}` | (modification time) |

Changing these settings parses the dataset again. Measurements with the same time are ordered by their file name. The
cache is rebuilt when the files or their order change.

## Grouping measurements for the queue

"Auto queue" (measurement selection, group "Queue actions") splits the dataset into selections and adds them to the queue.
Each selection is a group of sample measurements that is evaluated together, plus one reference per sample measurement.
Three settings next to the button control how the groups are formed and how references are assigned.

To check a queued selection, select it in "Selection queue" and click "Show selection details". A separate window lists:
- a summary: sample names and folders, measurement time span, positions, and the time and distance between each sample
  and its reference;
- every sample measurement with its position and its reference, including the time between them and the distance.

### Sample grouping

| Option | Groups |
|---|---|
| Consecutive at same position (default) | Uninterrupted runs of measurements at one position |
| Same position | All measurements at a position, also when other measurements were taken in between |
| Same name | All files with the same name, regardless of their coordinates |
| Same name and position | Like "Same name", but a name measured at two positions gives two groups |
| Each measurement | One selection per file |

### Name pattern

Used by "Same name" and "Same name and position" to determine the name of a file.

- **Empty (default):** the file name without the timestamp and the `X_… mm` / `Y_… mm` coordinates, e.g.
  `2026-07-16T14-53-01.011314-Pectin_avg_100-sample1_2-X_34.600 mm-Y_8.000 mm.txt` → `Pectin_avg_100-sample1_2`.
- **Regular expression:** searched in the file name (without extension). If it has a capture group, the first group is the
  name, otherwise the whole match. Files that don't match are not queued, so the pattern also works as a filter.
  - `sample(\d)_` puts all locations of a sample (`sample1_1`, `sample1_2`, ...) into one group.
  - `sample(\d_\d)` groups per location.

### Reference pairing

| Option | Reference for each sample measurement |
|---|---|
| Reference matching criterion (default) | From the nearest reference block if the reference matching criterion is "File selection", otherwise from that criterion |
| Nearest reference block | From the run of reference measurements closest in time. A group made of several runs gets the nearest block for each run |
| Nearest reference in time | The single reference measurement closest in time |
| Preceding reference | The last reference measured before it (the first following one if there is none) |
| Following reference | The first reference measured after it (the last preceding one if there is none) |

### Examples

**Repeated measurements at fixed positions** (e.g. 10 sample measurements at a location, then 10 references):
the defaults, "Consecutive at same position" with "Reference matching criterion".

**Area scan with alternating sample and reference measurements** (e.g. 10 points of a small area, a reference after each
sample point, all sample files share a name but have different coordinates):

- Sample grouping: "Same name"
- Reference pairing: "Following reference" if each reference is measured after its sample point, "Preceding reference" if
  it is measured before

This gives one selection per area with 10 samples and their own 10 references.

**All locations of a sample together:** Sample grouping "Same name" with the name pattern `sample(\d)_`.

The queue lists the selections after "Auto queue" ("Clear queue" removes them). Check the sample and reference counts in the
selection info before evaluating.

## How the q-space evaluation works

The q-space evaluation determines the thickness d and the complex refractive index n = n + ik of a sample from its
transmission (sample / reference spectrum). For a fixed thickness the transmission model can always be fitted, but with a
wrong thickness the extracted n still contains the Fabry-Pérot (FP) oscillations caused by the echoes inside the sample.
The q-value measures how strong these oscillations are; the thickness with the lowest q-value is the result.

For each evaluated measurement (or the average of a selection):

1. **Preparation**
   - The FP round trip time is estimated from the group delay of the transmission ("Estimate Fabry-Perot round trip
     time").
   - The number of echoes visible in the time trace follows from it ("Use estimated number of Fabry-Perot echoes"). The
     transmission model only includes these echoes.
   - The number of splines also follows from it ("Automatic spline count", "Splines per Fabry-Perot period"). n has to
     be able to follow the FP oscillations, otherwise q can't tell the thicknesses apart.
2. **Thickness scan:** one fit per thickness on the "Thickness axis bounds" with the "Custom thickness axis step" (and
   per time shift, if "Enable pulse shift optimization" is on). The fits run in parallel worker processes.
3. **Fit at a fixed thickness**
   - Initial guess for n ("Initial guess method"): the time of flight estimate, or a scan of constant n values with the
     transmission model ("Scanned n range", "Scan step n", "Maximum scanned k").
   - A smooth (regularized) spline curve for n and k is fitted to the measured transmission ("n regularization",
     "k regularization", at most "Maximum number of function evaluations").
   - With "Per frequency refinement", n is then solved exactly at each frequency, starting from the spline. This step
     determines the final n; the spline only has to start it on the right solution. A few spline evaluations (~5) give
     the same result as many and are much faster.
4. **q-value:**
   - The extracted n (or k / alpha, "Quantity used for q-value calculation") is Fourier transformed over frequency.
     FP oscillations show up as a peak at the FP round trip time, and q is the height of the spectrum there.
   - "Limit q-value range to visible echo" restricts this to the frequencies where the echoes are stronger than the
     "Echo amplitude threshold".
   - "Remove single-frequency spikes" (off by default) replaces isolated one-point spikes in that quantity (e.g. a
     per-frequency solution on another branch) by the mean of their neighbours before the transform. On the test sample
     there are no spikes inside the q-value band and the results are identical with and without it.
   - A timing drift of the THz pulse between sample and reference shifts the thickness by about 0.3 µm per fs: the
     phase fixes (n - 1)·d, the echo n·d. See "Reference drift" below.
5. **Optimum ("Optimum selection"):** the most prominent local minimum of q over the thickness, or the global minimum.
   With "Optimize thickness iteratively", the thickness grid is refined around the optimum ("Iteration count",
   "Thickness step shrink factor").

## Reference drift and stripes in images

The THz system drifts slowly in pulse timing and amplitude (e.g. tens of fs and ~1 % with a period of half an hour from
the lab temperature). If references are measured only now and then, e.g. at the start of each scan row, the drift
between reference and sample shows up as stripes along the scan lines, also in thickness and n. The corrections are
off by default and set in Settings:

- **Settings → Preprocessing → Reference drift:**
  - "Interpolate reference in time": the transmission uses the matching reference and the reference measured on the
    other side of the sample in time, interpolated to the sample's measurement time (amplitude and phase), if they are
    at most "Maximum time between interpolated references" apart.
  - "Median of reference blocks": references measured in a row (less than the block gap apart, e.g. the air spots of
    one opening of a sample holder) are combined into their median, so single partly blocked references at the edge of
    an opening don't set the amplitude.
- **Settings → Plotting → Image destriping** ("Remove stripes"): for image plot quantities that don't use a reference
  (e.g. peak to peak, pulse delay), or remaining stripes. The median of each line within the region is compared to the
  median of the neighbouring "Lines for the trend" lines and the difference is removed from the line ("Relative
  offsets": divided instead, for amplitudes). It assumes the sample in the region has no real line to line structure.

Example, sapphire plates in a holder with air openings at the row starts (rows of 10.6 min, drift ±40 fs / ±1 % with a
~35 min period): with a reference threshold that includes the air at the row starts, the row to row jitter on the
sample went from 5.6 fs / 0.9 % (nearest reference) to 3.5 fs / 0.24 % with interpolation and block median (22 fs /
0.6 % with references hours away). Destriping the raw pulse delay image with a 15 line trend reduced its stripes in the
region from 22 to 4.5 fs; parts of a row measured minutes later than the region keep more of the drift.

## Uncertainty

Settings → Evaluation → Uncertainty. The uncertainties are Monte Carlo estimates: the measured transmission is
perturbed randomly with its estimated uncertainty and evaluated again, and the spread of the results is the standard
uncertainty (1 σ) of the shown value.

Uncertainty of the transmission:
- **Measurement noise:** fitted to the differences of consecutive references (median, so references next to the
  holder edge don't count) as additive noise plus amplitude and timing jitter of each measurement. The fit is logged,
  e.g. "amplitude jitter 0.42 %, timing jitter 1.09 fs, additive noise 0.06". For averaged measurements the noise
  averages down. With 3 or more measurements their scatter is used instead, if it is larger. With the drift
  correction the reference noise is reduced by the averaging of the block median and the interpolation.
- **Reference drift** ("Include the reference drift"):
  - Without the drift correction: the change of the reference pulse (arrival time, amplitude) between the reference
    and the sample measurement, interpolated in the time series of all references.
  - With it: the residual error of the correction. Each reference block near the measurement is predicted from its
    neighbours, like a sample. Blocks of references that are clipped by the holder are left out for the amplitude:
    that clipping is the same in every line and isn't drift (a higher reference threshold avoids such references).
  - "Additional timing / amplitude uncertainty" are added for sources that aren't in the data, e.g. a sample holder
    that changes the beam path.

Where it is used:
- **Q-space thickness** ("Thickness uncertainty (Monte Carlo)", off by default): after the thickness scan of each
  measurement, every perturbed transmission ("Monte Carlo samples", 20) gets a scan of a thickness window around the
  optimum (± "Thickness window", "Thicknesses per window"). The window moves on if a sample's minimum is at its end.
  The spread of their q-value minima is the thickness uncertainty ("Thickness uncertainty" in the result view, also
  in the log); the spread of their n(f) is the uncertainty band of n and alpha at the optimum. It costs samples ×
  thicknesses fits per measurement (140 with the defaults, a few seconds). If the samples' mean differs from the
  optimum by more than the noise explains, the log says that the thickness step limits the optimum (use refinement
  iterations).
- **Image pixels:** "Thickness uncertainty of inspected pixel" and "Thickness uncertainty of area" (Image evaluation,
  group "Uncertainty") run the same for pixels with a result, shown as the quantity "Q-space thickness
  uncertainty". They are saved with the image results.
- **Conductivity** ("Conductivity uncertainty (Monte Carlo)", on, 200 samples): noise and drift of the film and the
  substrate measurements, "Film thickness uncertainty" and "Substrate thickness uncertainty" (Conductivity
  evaluation; with a q-space result of the substrate its Monte Carlo thickness uncertainty, if it has one; the
  substrate n changes with the thickness at constant optical thickness). The drift is drawn once for all film
  measurements, since measurements close in time share it. The log lists the contribution of each source to the
  uncertainty of the mean. The uncertainty of the average is the larger of its Monte Carlo uncertainty and the
  standard error of the mean of the measurements.

Example, sapphire image with the drift correction: the predicted thickness uncertainty of 35 pixels (median 0.97 µm)
matches their actual scatter (1.04 µm, including the residual stripes); without the correction a pixel 22 fs from its
reference got ± 4 µm. The conductivity of a ~9 S/cm film on it: ± 4 S/cm for the mean of 81 measurements, mostly from
the ~2-3 fs residual timing drift between the film and the single substrate measurement, which averaging more film
measurements doesn't reduce.

## Image evaluation

"Q-space evaluation → Image evaluation" (shown when "Image evaluation" is enabled in Settings → Component toggle →
Q-Space evaluation) runs the q-space evaluation for every pixel of the image grid. Each pixel is evaluated from the mean
transmission of its measurements, with the fit settings of the q-space evaluation. One spline count and echo count is
used for the whole image, so the q-values of the pixels are comparable.

The image evaluation tab holds what is set per image: the evaluated area and the mask ("Evaluated area"), "Run",
"Re-evaluate" and "Results". The tuning settings of the method below (neighbour guided scan, thickness search, outlier
re-evaluation, reliability, minimum time of flight, autosave interval) are in Settings → Evaluation → Image evaluation;
they rarely need changing.

Color scale of the images: pixels without a value (not evaluated, air, holder, no sample measurement) are transparent
and don't count for the color bar limits. By default the limits are the smallest and largest value. With
"Automatic color bar limits" (Settings tab, plot settings, group "Image") they are the percentiles "Automatic limits:
percentiles" (1 - 99 %) of the shown pixels, so single outliers don't stretch the scale, and with "Automatic limits:
centered on the median" (on) symmetric around the median (e.g. the median thickness). The limits used are logged.

To save time it uses the fact that the thickness changes little from pixel to pixel:

1. **Pixels without sample are skipped**
   - Pixels whose time of flight is below "Minimum time of flight" are air.
   - With "Skip pixels outside a value range", pixels whose value of the "Mask quantity" (e.g. peak to peak for the
     opaque holder) is outside the "Mask value range" are skipped as well. The values are those of the image plot, so
     the range can be read off the image.
   - "Shrink the sample area (pixels)" removes that many pixel rings next to the skipped pixels, e.g. the pixels at the
     holder edge that are only partly covered by the sample.
   - Select "Q-space sample mask" in the image plot to see which pixels would be evaluated (1) or skipped (0).
2. **Probe pixels:** a few random pixels ("Probe pixels") get the full thickness scan. All later full scans only cover
   the thickness range of their results plus the "Probe range margin". If an optimum lands at the edge of that range, the
   full configured range is scanned for that pixel.
3. **Coarse to fine**
   - Every "Seed pixel spacing"-th pixel in x and y gets a full thickness scan.
   - Then the spacing is halved, and each new pixel only scans a narrow window ("Thickness window around neighbours",
     "Thickness window step") around the median thickness of its already evaluated neighbours.
   - If the optimum lands at the edge of the window, the window is moved there ("Maximum window moves").
   - If the q-value is much higher than the neighbours' ("Maximum q-value ratio to neighbours", e.g. at a thickness
     step), the pixel gets the full scan.
4. **Interpolated minimum:** with "Interpolate the q-value minimum", a parabola through the best scanned thickness and its
   two neighbours gives the thickness of the minimum, followed by one fit at that thickness. This replaces the
   refinement iterations.
5. **Outliers:** pixels whose thickness deviates from their 8 neighbours by more than the "Thickness deviation
   threshold" (and more than 3× the neighbours' scatter) are scanned again around the consistent neighbours. The new
   result is only used if it is trusted, so real thickness steps keep their value. "Re-evaluate outliers" repeats this
   step after the evaluation.

"Pixel binning" n evaluates blocks of n × n pixels as one pixel (mean transmission), for a quick preview or noisy data
(2: a quarter of the pixels).

The results are available in the image plot as
- "Q-space thickness", "Q-space q-value", "Q-space time shift";
- "Q-space refractive index" and "Q-space absorption coe." at the selected frequency;
- "Q-space refractive index (band mean)" and "Q-space absorption coe. (band mean)": the mean over "Band of the band mean
  quantities" (equal bounds: the whole fit range);
- "Q-space full scan" (1: full thickness scan, 0: neighbour window);
- the reliability values below.

"Thickness statistics" shows a histogram of the thickness (all and reliable pixels) and the mean, median, standard
deviation and 5-95 % range of the reliable pixels.

"Save image results" writes everything to an `.npz` file in the result folder:
- per pixel: x, y, d, shift, q, full scan, the reliability values, n over frequency, the q(d) curve, measurement names;
- the skipped pixels and what continuing or re-evaluating needs (echo and spline count, thickness range, binning);
- the settings of the q-space evaluation and the global settings (`settings`, JSON) and the code version (`git
  describe`).

Neighbouring pixels are only used to steer the thickness scan, every pixel still gets its own thickness. The pixel to pixel
differences in the thickness are real: predicting a pixel from its neighbours is typically off by ~3 µm, while repeated
evaluations of the same pixel agree to ~0.4 µm. For the same reason the thickness can't be fitted with one shared n(f)
for the whole image: n varies by ~0.02 from pixel to pixel, which shifts a thickness fitted with a fixed n by ~20 µm.

### Reliability

Every pixel result gets a reliability rating (settings in Settings → Evaluation → Image evaluation, group "Reliability and inspection"), available as image plot quantities:

- **Q-space minimum width:** the distance from the minimum at which a parabola through the optimum and up to two
  scanned thicknesses on each side doubles the q-value. A sharp minimum pins the thickness down. The natural width
  depends on the sample and the bandwidth (~25 µm here), so it is compared to the median width of the image.
- **Q-space q-value / neighbours:** the q-value divided by the median q-value of the 8 neighbours.
- **Q-space thickness - neighbours:** the thickness minus the median thickness of the 8 neighbours.
- **Q-space reliable:** 1 if the minimum width is at most "Maximum width of the q-value minimum / median" times the
  median width, the q-value ratio is at most "Maximum q-value ratio to neighbours" and the thickness deviation is at most
  the outlier limit (the "Thickness deviation threshold", or 3× the neighbours' scatter). Pixels with fewer than 3
  neighbours are only rated by their minimum width.

"Evaluated pixels" shows the number of reliable pixels after a run. On the test sample, 93% of a smooth area are reliable,
70-77% of areas with thickness steps or at the holder edge and the sample border. The unreliable pixels are the
partially covered ones, those whose q(d) has no clear minimum and isolated thickness jumps.

### Inspecting and re-evaluating pixels

With "Inspect pixel on click", clicking a pixel in the image plot of a Q-space quantity plots, instead of the usual
on-click plot:

- its q(d) curve with the chosen optimum, the fitted parabola, the neighbours' thicknesses and the allowed deviation;
- n and alpha of the pixel and its neighbours.

The title shows all reliability values.

- **Re-evaluate inspected pixel:** a full scan of the configured thickness range for the last clicked pixel. The plot
  updates when it's done.
- **Re-evaluate area:** evaluates the pixels in "x-range" / "y-range" of "Evaluated area" again (also when "Confine to
  area" is off), with the same steps as the image evaluation. Pixels at the area border get window scans around their
  neighbours outside the area, the rest of the image is kept.

Both use the current settings, except the echo and spline count of the image evaluation, so that the q-values stay
comparable to the other pixels. A changed fit range requires a new image evaluation.

### Resuming and loading results

During the evaluation the results are saved to `q_space_image_autosave.npz` in the result folder every "Autosave
interval" and when the run ends, also when it is cancelled or fails.

- **Continue evaluation:** evaluates the pixels without a result (of the image, or of the area with "Confine to area"),
  e.g. after "Cancel". The already evaluated pixels guide the window scans like in the full run. On the test sample a
  cancelled and continued run gives the same thicknesses as an uninterrupted one, apart from a few ambiguous pixels at
  the holder edge.
- **Load image results:** loads the "Results file", or the newest q-space image result in the result folder if no file
  is selected (e.g. the autosave after a crash). The fit range must be the same as when the results were made. Then
  "Continue evaluation", the re-evaluations, inspecting pixels and saving work as after a normal run.

Continuing and re-evaluating evaluate an image area by adding to the existing results, so an image can also be done
area by area.

### Progress

"Overall progress" in the processing group of the q-space evaluation covers the whole run (the whole queue, or the whole
image), "Time remaining" estimates the remaining time and the finishing time from the fit rate so far. For a queue the
number of fits is known in advance. For an image it depends on how many window moves and full scan fallbacks are needed,
so the total starts as an estimate and is corrected while the evaluation runs (most accurate after the first levels).

### Speed

Almost all of the time is spent in the fits (per pixel: ~6 fits in the neighbour window, ~10-20 for a full scan). The
settings with the largest effect:

- "Maximum number of function evaluations" ~5 with "Per frequency refinement" on (same results as 20, several times
  faster).
- "Trust region solver" "Iterative (LSMR)" instead of "Exact (SVD)": cheaper spline fit steps, same thickness and q-value
  with "Per frequency refinement" on.
- "Use compiled kernels (numba)" (on by default): compiled versions of the transmission model, the initial guess scan and
  the per frequency refinement for the single layer model `tmm_1layer`, same results, ~1.5-2x faster fits. Needs the
  `numba` package; without it, or for other models, the normal code is used. The first run compiles the kernels (a few
  seconds per worker), later runs load them from the cache.
- "Scanned n range" limited to the plausible range of the sample, e.g. 1-2.5 for wood instead of 1-6.
- "Iteration count (window scan)" 1 with the interpolated minimum.
- Skipping the holder with the value mask: such pixels mostly end up in full scans.
- Test the settings on a small area first ("Confine to area", "x-range", "y-range"), or on the binned image
  ("Pixel binning" 2).

## Multi-layer samples

The q-space evaluation can fit one unknown layer inside a stack of known layers. n, k and the thickness of the unknown
layer are determined as for a single layer (thickness with the lowest q-value); the known layers enter the
transmission model with fixed refractive index and thickness.

1. Select the transmission model `tmm_multilayer` in the q-space evaluation.
2. Build the stack in "Q-space evaluation → Multi-layer sample" (shown when "Multi-layer sample" is enabled in
   Settings → Component toggle → Q-Space evaluation):
   - Choose the layer type, fill in the fields and press "Add layer". The new layer is inserted below the selected
     layer, or at the end if none is selected.
   - **Known, from result file:** a saved q-space result (HDF5) of the layer material. Its optimum result gives n and
     k, the thickness is filled in from the result and can be changed.
   - **Known, constant n:** constant n and k with the given thickness.
   - **Unknown (fitted):** the layer to evaluate. Exactly one is needed and only one can be added.
   - Select a layer in the list to load it into the editor, "Update selected layer" applies the changes. The buttons
     below the list remove and reorder layers.
3. Set the wave direction. The list shows where the THz wave enters (incident medium) and the arrows show its direction
   through the layers. "Bottom to top" reverses the order in which the layers are passed.
4. Run the evaluation as usual. The thickness axis bounds apply to the unknown layer.

Notes:

- With air on both sides the transmission is the same in both directions, so the wave direction does not change the
  result; it documents the measurement geometry.
- Outside the frequency range of a result file its n and k are held constant (a warning is logged).
- The Fabry-Pérot estimates (round trip time, echo amplitude and the q-value frequency range) are corrected for the
  known layers. A known neighbour with a refractive index close to the unknown layer gives weak echoes in the unknown
  layer, so the automatic q-value range is shorter and the thickness is less well determined.
- The time of flight initial guess does not account for the known layers, the model cost scan is used instead.
- The layer stack is saved in the config file and in each result file (`layer_stack` in the optimization info).

From the command line the stack is set as JSON (types `unknown`, `result`, `constant`):

```
python cli.py --set q_space_eval.transmission_model=tmm_multilayer \
              --set 'q_space_eval.layer_stack.layers=[{"type": "result", "label": "substrate", "path": "/path/result.hdf5", "d": 500}, {"type": "unknown", "label": "film"}]' \
              qspace --out results.csv
```

## Additional views and tools

Each of these can be switched off on its own: the views and the effective medium tab in Settings → Component toggle
("B-scan / C-scan", "Spectral explorer" under Image plot, "Spectrogram" under Dataset plots, "Effective medium"), the
absorption limit in Settings → Evaluation → Dynamic range, the image sharpening in the plot settings (group "Image
sharpening", off by default).

### B-scan / C-scan (Image plot)

- **B-scan**: the time traces along a line as an image, position along the line against delay. Echoes and layer
  interfaces of the sample are horizontal lines, steps in the thickness shift them. The line is the one of the slice
  plots ("Use the slice plot line") or its own. "Envelope (dB)" shows the envelope (magnitude of the analytic signal)
  relative to the maximum, so the echoes are visible independent of their sign.
- **C-scan**: an image at one delay, with a slider for the delay (depth). "Shown value": the field at the delay, the
  envelope maximum in a gate (gate width), or the delay of the envelope maximum in the gate (after the gate start), a
  depth map of a reflection or echo. Below the image the mean trace with the gate. "Delay range" limits the traces
  (all traces of the image are kept in memory: e.g. 7777 x 4001 delays = 124 MB, about 2 s).

### Spectral explorer (Image plot)

Uses the selected quantity of the image plot, which has to be a spectrum (e.g. amplitude transmission, absorbance,
phase, Q-space refractive index / absorption), on the frequency range of the image grid calculation.
- **Frequency explorer**: the image with a slider over the frequencies, and the median spectrum (with the 10 - 90 %
  band of the pixels); a click on a pixel adds its spectrum.
- **False-color image**: the band means of three frequency bands as red, green and blue. Each band is stretched to its
  1 - 99 % range, or one scale for all three ("Common scale for the bands"). Regions of different material show up in
  different colors even where the quantity at a single frequency is similar.

Both take as long as an image of the quantity (most of it is finding the reference of every measurement).

### Spectrogram (Dataset plots)

"Spectrogram of the selection": when which frequencies arrive, for the (mean of the) selected sample measurements and
their references. Short-time Fourier transform with a Gaussian window ("Window width": longer gives a finer frequency
resolution and a coarser time resolution) or Morlet wavelets ("Wavelet cycles", the same trade-off; at low frequencies
coarser in time). Fabry-Perot echoes repeat the pulse later, water vapour lines ring long after it at their
frequencies, dispersion tilts the arrival time with frequency.

### Maximum measurable absorption (Settings → Evaluation → Dynamic range)

Where the sample attenuates the reference down to the noise floor, the extracted absorption coefficient no longer
follows the sample: alpha_max d = 2 ln(DR 4n / (n+1)^2) (Jepsen & Fischer, Opt. Lett. 30, 29 (2005)), with DR the
reference amplitude spectrum over the noise floor (the additive noise of the noise model, see Uncertainty, divided by
the square root of the number of averaged measurements). The q-space results get the quantity "alpha_max" (check it
together with alpha in the result view), the log says from which frequency on alpha reaches 90 % of it, and the pixel
view of the image evaluation shows it (with its smallest value in the legend, as it is often far above alpha).

The smallest measurable absorption ("Minimum measurable absorption", "Coverage factor (alpha_min)"): alpha =
-2/d ln(|T| / t_Fresnel), so the uncertainty of ln|T| gives alpha an uncertainty of 2/d sigma(ln|T|), whatever alpha
is. alpha_min is the coverage factor (default 2) times that, with sigma(ln|T|)^2 = E|dT|^2 / (2|T|^2) + a^2 from the
transmission noise (noise model or the scatter of the averaged measurements) and the relative reference amplitude
uncertainty a (drift and "extra amplitude uncertainty", Settings → Evaluation → Uncertainty). Below it alpha can't be
told apart from 0. The q-space results get the quantity "alpha_min", the log says at which frequencies alpha is below
it, and the pixel view of the image evaluation shades the range below it. Systematic errors (Fresnel losses of a
wrong n, scattering, a model that doesn't fit the sample) aren't included.

### Image sharpening (plot settings, group "Image sharpening")

Deconvolves the shown image with the beam profile, a Gaussian intensity profile with the 1/e^2 radius "Beam radius" or,
with "Use the knife-edge radius", the beam radius of the last knife-edge fit (slice plots, integrated power). Pixels
without a value are left out. Richardson-Lucy (iterations: more is sharper) or a Wiener filter (signal to noise ratio:
larger is sharper). Exact for quantities linear in the transmission, approximate for derived ones (thickness, n). With
too large a radius the values overshoot and ring at edges: on the sapphire image (0.5 mm pixels, edges within ~1 mm) a
radius of 0.5 mm sharpens slightly, 1 mm already rings.

### Effective medium (own tab)

Mixtures of a host (matrix, e.g. PE powder, or air for porosity) and inclusions with volume fraction f. Models:
Bruggeman (both phases equivalent, any fraction), Maxwell Garnett (dilute spherical inclusions), Landau-Lifshitz-
Looyenga (shape independent, powders and porosity) and linear in n (time of flight adds up). Host and inclusion index:
a constant n, k or the optimum result of a q-space result file. "Calculate":
- **Volume fraction from the mixture**: fits f (0 - 1) to the mixture index of a q-space result file over the frequency
  range. At 0 or 1 the measured index is outside of what the model gives with these components.
- **Inclusion index from the mixture**: the index of the inclusions at the given fraction (e.g. a substance in a PE
  pellet with known mass and density).
- **Mixture index from the fraction**: the model prediction.

Image quantity "Effective medium volume fraction": the fitted fraction of every pixel of the q-space image evaluation
(its n(f) over the frequency range), with the host and inclusion settings of the tab.

## Remote workers

The fits of the q-space evaluation (selections, image evaluation, Monte Carlo uncertainty) can run on another
computer, e.g. a Linux PC with more cores, while the data, the GUI and the results stay on this computer. Each task
only carries the transmission of one pixel or measurement and the settings (a few kB); the measurement files are never
copied. Settings: Settings tab, "Remote workers".

How it works: "Connect" starts one ssh session that forwards a local port to the remote and starts the worker server
there. The Python files of this repository are sent along (cached in `~/.teval_remote/code/` on the remote), so the
remote always runs exactly the local code, uncommitted changes included, and needs no copy of the repository. The
server listens on a Unix socket in `/tmp` that only your account can open and only accepts this session's random key.
It stops when you disconnect, close the program or the ssh connection drops (e.g. the VPN). If the connection is lost
during an evaluation, the local workers finish it.

One-time setup:
1. Login without a password: generate a key, send it to the remote (this asks for the password of the teval account
   once) and check the login.

   Windows (PowerShell):
   ```
   ssh-keygen -t ed25519
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh teval@pc13203.physik.uni-marburg.de "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ssh teval@pc13203.physik.uni-marburg.de echo ok
   ```

   Linux / macOS (terminal):
   ```
   ssh-keygen -t ed25519
   ssh-copy-id -i ~/.ssh/id_ed25519.pub teval@pc13203.physik.uni-marburg.de
   ssh teval@pc13203.physik.uni-marburg.de echo ok
   ```

   The last command must print `ok` without asking for a password.
   - If you already have a key, `ssh-keygen` asks `Overwrite (y/n)?`: answer `n` and use the existing key (skip the
     first command). Overwriting it locks you out wherever that key is used (e.g. GitHub).
   - The second command appends your key to the remote's `~/.ssh/authorized_keys` (`>>`, `ssh-copy-id` does the
     same), the keys of the others stay. Each line there is one key and ends with a comment like `name@computer`;
     deleting the line removes that person's access.
   - If you gave the key a passphrase, load it into the ssh agent once per login, otherwise the program can't use the
     key (it can't ask for the passphrase). Windows: start the service "OpenSSH Authentication Agent" (Services,
     startup type "Automatic"), then `ssh-add`. Linux / macOS: `ssh-add` (desktop sessions usually run an agent
     already; otherwise first `eval "$(ssh-agent)"`).
2. Settings, "Remote workers": "Host" is `teval@pc13203.physik.uni-marburg.de` by default (the account made for
   the remote workers), the same inside the institute network and over the VPN. A host alias of `~/.ssh/config`
   works as well (e.g. for a jump host).
3. "Set up the Python environment on the remote": creates your own environment `~/.teval_remote/venv-<your user
   name on this computer>` on the remote with the packages of `requirements.txt`, in the versions installed on this
   computer (under a minute when the packages are already in the remote's pip cache, otherwise a few minutes; the
   output appears in the message panel; run it again after updating packages here). It needs Python 3.11 or newer
   ("Python for the setup") with the `venv` module (on Debian / Ubuntu the package `python3-venv`). Alternatively
   set "Python on the remote" to an existing Python environment with the requirements (empty: your own environment).

Use: "Connect" (or "Connect on start"). The message panel shows the remote's name, worker count, Python version and
load. Evaluations then use the remote workers plus the local ones ("Use the local workers too", "Number of cpu cores to
assign" of the evaluation). "Remote workers" 0 uses all cores of the remote; leave some free if others work on it.

Sharing the remote account: several people can be connected at the same time, each connection has its own server,
socket and key, and everyone has an own Python environment (so one person's setup doesn't change the package versions
of the others). The code is cached per version, so different code versions don't interfere either. The cores are
shared, though: with "Remote workers" 0 every connection starts a worker per core, so if you expect to evaluate at the
same time, agree on e.g. 10 workers each.

What to expect (sapphire image, 1681 pixels, local AMD Ryzen 7 9800X3D, remote Intel Xeon W-2255 with 10 cores / 20
threads): 6 local workers 144 s, 6 local + 20 remote 96 s, 20 remote only 212 s. A remote core is about half as fast
as a core of a recent desktop CPU, so the gain depends on the two computers; "Remote workers" only is still useful to
keep this computer free.

The single fits agree to ~1e-15 between the computers, but different CPUs round differently. In about 1-2 % of the
pixels (those with two nearly equal q-value minima, or a value at a decision threshold) that changes the chosen
thickness by up to a few µm, and through the neighbour windows also nearby pixels. Which tasks run where varies from
run to run, so with remote workers these pixels can differ between runs; with the local workers only a run is exactly
reproducible. The setup installs the local package versions on the remote (if pip has them for its Python), so that
the fits behave the same otherwise.

Command line: `python cli.py --remote qspace ...` connects before the evaluation (with the host etc. of the settings).

Troubleshooting:
- "Connection failed": the message panel has the ssh / server error. Check
  `ssh teval@pc13203.physik.uni-marburg.de echo ok` first. "The Python environment doesn't exist on the remote": run
  the setup step 3.
- "Python package ... missing on the remote": run the setup step 3, or install the requirements in the configured
  Python.
- If the ssh client can't forward to a Unix socket, set "Remote port" to a free port number (e.g. 47000): the server
  then listens on that TCP port of the remote's localhost instead.
- Each result costs this computer roughly 2 ms of processing; with very fast fits (~30 ms) and many workers in total
  that can become the limit (the main process at 100 % of one core).

## Command line interface

`cli.py` runs the evaluations without the GUI. It uses the same settings file as the GUI (or `--config <file>`); changes made
with `--set` only apply to that run unless `--save-settings` is given.

```
python cli.py selections                                  # list the auto queued selections
python cli.py qspace --out results.csv                    # q-space thickness evaluation of all selections
python cli.py qspace --select sample1_2 --no-save         # only selections with "sample1_2" in a sample file name
python cli.py --data /path/to/dataset qspace              # another dataset directory
python cli.py get dataset.measurement_selector            # show settings and their names
```

Settings are addressed by their dotted path as shown by `get`:

```
python cli.py --set dataset.measurement_selector.sample_grouping=name \
              --set dataset.measurement_selector.queue_ref_pairing=following \
              --set "q_space_eval.d_opt_axis_bounds=200, 500" \
              qspace --out results.csv
```

- Quantities use the setting's unit unless one is given (`300 um`).
- Ranges are given as `start, end`.
- Options are given by name (see `get` or the error message for the valid names).

The `qspace` command prints the optimum thickness per selection, writes it to a CSV file with `--out`, and saves the HDF5
result files like the GUI (disable with `--no-save`). `--gui-selection` evaluates the measurement selection stored in the
config instead of the auto queued selections. Log messages go to stderr, the result table to stdout.
