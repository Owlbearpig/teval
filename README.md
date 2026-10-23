# teval

Evaluation tool for THz time-domain spectroscopy (THz-TDS) measurements, with a GUI that is generated automatically from the
evaluation components.

## Starting the GUI

Run from the repository root:

```
python load_with_ui.py eval_scripts/test_script.py
```

Settings are loaded from and saved to `config/<script name>.json`, e.g. `config/test_script.json`, when the window is closed.

## Grouping measurements for the queue

"Auto queue" (measurement selection, group "Queue actions") splits the dataset into selections and adds them to the queue.
Each selection is a group of sample measurements that is evaluated together, plus one reference per sample measurement.
Three settings next to the button control how the groups are formed and how references are assigned.

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

## Command line interface

`cli.py` runs the evaluations without the GUI. It uses the same settings file as the GUI; changes made with `--set` only apply
to that run unless `--save-settings` is given.

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
