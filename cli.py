"""Command line interface for the teval components, without the GUI.

Examples:
    python cli.py selections
    python cli.py --data /path/to/dataset qspace --select sample1_2 --out results.csv
    python cli.py --set settings.eval_opt.fp_count=10 --set "q_space_eval.d_opt_axis_bounds=200, 300" qspace
    python cli.py get settings.eval_opt
"""
import argparse
import contextlib
import csv
import json
import logging
import os
import sys
import time
from enum import Enum
from os.path import basename, splitext
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from traitlets import Instance, List
from common.components import ComponentBase
from common.settings import Settings
from common.remote import client as remote_client
from common.eval_component.q_space_eval import OptimizationCancelled
from common.traits import ValueRange
from common.units import Q_

log = logging.getLogger("cli")


class CliError(Exception):
    pass


class ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.count = 0

    def emit(self, record):
        self.count += 1


def load_app_root(script):
    ComponentBase.script_name = script
    script_globals = {"__name__": splitext(basename(script))[0]}
    exec(compile(open(script, "rb").read(), script, "exec"), script_globals)
    return script_globals["AppRoot"]()


def resolve(root, path):
    obj, parent, name = root, None, None
    for part in path.split(".") if path else []:
        if not isinstance(obj, ComponentBase) or part not in obj.traits():
            options = ", ".join(sorted(obj.traits())) if isinstance(obj, ComponentBase) else ""
            raise CliError(f"'{part}' not found in '{path}'. Options: {options}")
        parent, name, obj = obj, part, getattr(obj, part)
    return parent, name, obj


def is_component(trait):
    return isinstance(trait, Instance) and isinstance(trait.klass, type) and issubclass(trait.klass, ComponentBase)


def parse_scalar(text, current):
    text = text.strip()
    if isinstance(current, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise CliError(f"Expected a boolean, got '{text}'")
    if isinstance(current, Enum):
        for member in type(current):
            if text in (member.name, str(member.value)):
                return member
        options = ", ".join(m.name for m in type(current))
        raise CliError(f"Unknown option '{text}'. Options: {options}")
    if isinstance(current, Q_):
        try:
            return Q_(float(text), current.units)
        except ValueError:
            return Q_(text).to(current.units)
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, Path):
        return Path(text)
    return text


def set_value(root, assignment):
    path, sep, text = assignment.partition("=")
    if not sep:
        raise CliError(f"Expected path=value, got '{assignment}'")
    parent, name, current = resolve(root, path.strip())
    if parent is None:
        raise CliError("Cannot set the root component")
    trait = parent.traits()[name]
    if trait.read_only or is_component(trait):
        raise CliError(f"'{path}' cannot be set")

    if isinstance(trait, List):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            raise CliError(f"'{path}' is a list, expected JSON: {e}")
    elif isinstance(trait, ValueRange):
        parts = text.split(",")
        if len(parts) != 2:
            raise CliError(f"'{path}' is a range, expected 'start, end'")
        value = [parse_scalar(p, c) for p, c in zip(parts, current)]
    else:
        value = parse_scalar(text, current)

    setattr(parent, name, value)
    log.info(f"Set {path.strip()} = {format_value(getattr(parent, name))}")


def format_value(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{format_value(value[0])}, {format_value(value[1])}"
    if isinstance(value, Q_):
        return f"{value:~P}"
    return str(value)


def print_component(component, path):
    for name, trait in sorted(component.traits().items()):
        full_name = f"{path}.{name}" if path else name
        if is_component(trait):
            if getattr(component, name) is not None:
                print(f"{full_name}  [component]")
            continue
        label = trait.metadata.get("name", "")
        flags = " (read only)" if trait.read_only else ""
        try:
            value = format_value(getattr(component, name))
        except Exception:
            value = "?"
        value = value if len(value) < 80 else value[:77] + "..."
        print(f"{full_name} = {value}{flags}" + (f"    # {label}" if label else ""))


def wait_until(app, condition, timeout, poll=0.02):
    t0 = time.monotonic()
    while not condition():
        app.processEvents()
        if timeout is not None and time.monotonic() - t0 > timeout:
            return False
        time.sleep(poll)
    app.processEvents()
    return True


def wait_for_parse(app, dataset, prev_cache, errors, timeout):
    error_cnt = errors.count
    t0 = time.monotonic()

    def done():
        if dataset.cache is not prev_cache and not dataset._parse_lock.locked():
            return True
        started_late = time.monotonic() - t0 > 2.0
        return started_late and not dataset._parse_lock.locked() and errors.count > error_cnt

    if not wait_until(app, done, timeout):
        raise CliError(f"Timed out loading dataset {dataset.data_path}")
    wait_until(app, lambda: dataset.is_initialized and len(dataset.measurements["all"]) > 0, 5.0)
    if dataset.cache is prev_cache or not dataset.is_initialized:
        raise CliError(f"Could not load dataset {dataset.data_path}")


def load_dataset(app, dataset, data_path, errors, timeout):
    config_path = Path(dataset.data_path)
    target = Path(data_path).expanduser().resolve() if data_path else config_path
    if str(target) in ("", "."):
        raise CliError("No dataset path in the config, pass --data")
    if not target.is_dir():
        raise CliError(f"Dataset path {target} is not a directory")

    if str(config_path) not in ("", ".") and config_path.is_dir():
        try:
            wait_for_parse(app, dataset, None, errors, timeout)
        except CliError:
            if target == config_path:
                raise

    if target != config_path or dataset.cache is None:
        prev_cache = dataset.cache
        dataset.data_path = target
        wait_for_parse(app, dataset, prev_cache, errors, timeout)

    log.info(f"Loaded {len(dataset.measurements['sams'])} sample and "
             f"{len(dataset.measurements['refs'])} reference measurements from {dataset.data_path}")


def queued_selections(dataset, select_strings, use_gui_selection):
    selector = dataset.measurement_selector
    if use_gui_selection:
        selection = selector.get_selection()
        if selection is None:
            raise CliError("The selection set in the config does not resolve to measurements with references")
        return [selection]

    queue = selector.selection_queue
    queue.clear()
    selector.auto_queue()
    selections = []
    while len(queue):
        selections.append(queue.pop_next()[1])

    if select_strings:
        selections = [sel for sel in selections
                      if any(s in m.filepath.name for m in sel.sams for s in select_strings)]
    if not selections:
        raise CliError("No sample measurement blocks match the selection")
    return selections


def cmd_selections(app, root, args):
    selections = queued_selections(root.dataset, args.select, args.gui_selection)
    for idx, sel in enumerate(selections):
        refs = sorted(set(sel.refs), key=lambda m: m.meas_time)
        print(f"{idx:3d}  {len(sel.sams):3d} samples  {len(refs):3d} references  "
              f"{sel.sams[0].filepath.name}")


def magnitude(value, unit=""):
    if value is None:
        return float("nan")
    return float(Q_(value).to(unit).magnitude) if isinstance(value, Q_) else float(value)


def optimum_rows(eval_data, selection_label, short_name):
    rows = []
    for meas_name, key in eval_data.optimum_results.items():
        res = eval_data.results[key]
        info = res.optimization_info
        rows.append({
            "name": short_name,
            "selection": selection_label,
            "measurement": meas_name,
            "d_um": magnitude(res.d, "µm"),
            "shift_fs": magnitude(res.shift, "fs"),
            "q_val": magnitude(info.get("q_val")),
            "gof": magnitude(info.get("gof")),
            "converged": bool(info.get("converged", False)),
            "fp_time_ps": magnitude(info.get("fp_time"), "ps"),
            "tested_d_cnt": len({round(r.d.magnitude, 6) for r in eval_data.results.values()
                                 if r.measurement == meas_name}),
        })
    return rows


def cmd_qspace(app, root, args):
    evaluator = root.q_space_eval
    if evaluator is None:
        raise CliError("The q-space evaluation component is not enabled in this script")
    if args.no_save:
        evaluator.result_saver.enabled = False

    selections = queued_selections(root.dataset, args.select, args.gui_selection)
    log.info(f"Evaluating {len(selections)} selection{'s' * (len(selections) != 1)}")

    rows = []
    t0 = time.monotonic()
    try:
        for idx, selection in enumerate(selections):
            log.info(f"[{idx + 1}/{len(selections)}] {selection.label}")
            eval_data = evaluator.eval_selection(selection)
            evaluator._result_ready(eval_data)
            short_name = evaluator.result_saver._get_main_file_name(eval_data.measurement_names)
            new_rows = optimum_rows(eval_data, selection.label, short_name)
            rows.extend(new_rows)
            for row in new_rows:
                log.info(f"  {row['measurement']}: d = {row['d_um']:.2f} µm, q = {row['q_val']:.4g}")
    except (KeyboardInterrupt, OptimizationCancelled):
        log.warning("Evaluation interrupted, writing the finished results")

    log.info(f"Finished in {time.monotonic() - t0:.0f} s")
    if not rows:
        return

    columns = list(rows[0])
    if args.out:
        with open(args.out, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Wrote {args.out}")

    name_width = max(12, *(len(row["name"]) for row in rows))
    print(f"{'measurement':<{name_width}} {'d (µm)':>9} {'shift (fs)':>10} {'q':>11} {'GoF':>10} {'conv.':>6}")
    for row in rows:
        print(f"{row['name']:<{name_width}} {row['d_um']:9.2f} {row['shift_fs']:10.1f} "
              f"{row['q_val']:11.4g} {row['gof']:10.3g} {str(row['converged']):>6}")


def cmd_get(app, root, args):
    parent, name, obj = resolve(root, args.path)
    if isinstance(obj, ComponentBase):
        print_component(obj, args.path)
    else:
        print(format_value(obj))


def build_parser():
    parser = argparse.ArgumentParser(description="teval command line interface",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--script", default="eval_scripts/test_script.py",
                        help="Entry script with the AppRoot class; settings come from config/<script name>.json "
                             "unless --config or another file chosen in the GUI")
    parser.add_argument("--config", help="Settings file (default: the one used last in the GUI, see Settings > "
                                             "Settings file)")
    parser.add_argument("--data", help="Dataset directory (default: the path saved in the config)")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="Change a setting for this run, e.g. settings.eval_opt.fp_count=10. "
                             "Quantities take the trait's unit unless one is given ('300 um'), "
                             "ranges are 'start, end', options are given by name. Repeatable")
    parser.add_argument("--save-settings", action="store_true",
                        help="Write the settings (including --set changes) back to the config file")
    parser.add_argument("--timeout", type=float, default=600, help="Dataset loading timeout in s")
    parser.add_argument("--remote", action="store_true",
                        help="Use the remote workers (Settings > Remote workers), also without \"Connect on start\"")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection_args(p):
        p.add_argument("--select", action="append", default=[], metavar="STRING",
                       help="Only blocks with a sample filename containing STRING. Repeatable")
        p.add_argument("--gui-selection", action="store_true",
                       help="Use the measurement selection stored in the config instead of auto queued blocks")

    p_sel = sub.add_parser("selections", help="List the sample measurement blocks and their references")
    add_selection_args(p_sel)
    p_sel.set_defaults(func=cmd_selections, needs_data=True)

    p_q = sub.add_parser("qspace", help="Run the q-space thickness evaluation")
    add_selection_args(p_q)
    p_q.add_argument("--out", help="Write the optimum per measurement to this CSV file")
    p_q.add_argument("--no-save", action="store_true", help="Don't save the HDF5 result files")
    p_q.set_defaults(func=cmd_qspace, needs_data=True)

    p_get = sub.add_parser("get", help="Show a setting or all settings of a component (dotted path)")
    p_get.add_argument("path", nargs="?", default="")
    p_get.set_defaults(func=cmd_get, needs_data=False)

    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.captureWarnings(True)
    errors = ErrorCounter()
    logging.getLogger().addHandler(errors)

    app = QtWidgets.QApplication(sys.argv[:1])

    with contextlib.redirect_stdout(sys.stderr):
        Settings.settings_file_override = args.config
        root = load_app_root(args.script)
        root.__enter__()

    try:
        if args.needs_data or args.data:
            with contextlib.redirect_stdout(sys.stderr):
                load_dataset(app, root.dataset, args.data, errors, args.timeout)

        for assignment in args.set:
            set_value(root, assignment)
        app.processEvents()

        if args.remote or root.settings.remote.connect_on_start:
            if root.settings.remote.connect_now() is None and args.remote:
                raise CliError("Could not connect to the remote workers")

        args.func(app, root, args)
    except CliError as e:
        log.error(str(e))
        return 1
    finally:
        remote_client.disconnect()
        if args.save_settings:
            with contextlib.redirect_stdout(sys.stderr):
                root.__exit__(None, None, None)
            log.info("Saved settings")

    return 0


if __name__ == "__main__":
    exit_code = main()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
