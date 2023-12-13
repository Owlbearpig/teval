from pathlib import Path
from scipy.constants import c as c0
from numpy import pi
import os
from os import name as os_name

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os_name:
    base_dir = Path(r"/home/ftpuser/ftp/Data/")
    data_dir = base_dir / "PolymerSamples_Exipnos" / "Meas0"
    result_dir = Path(r"/home/alex/MEGA/AG/Projects/Material characterization/Exipnos Samples/Results")
else:
    data_dir = Path(r"")
    result_dir = Path(r"")

try:
    os.scandir(data_dir)
except FileNotFoundError as e:
    raise e

post_process_config = {"sub_offset": True, "en_windowing": False}

# physical constants
THz = 1e12
d_msla = 1008  # um
angle_in = 0 * pi / 180
c_thz = c0 * 1e-6  # um / ps -> 1e6 / 1e-12 = 1e-6

# optimization constants
initial_shgo_iters = 3
shgo_bounds = [(1.5, 1.7), (0.01, 0.10)]

# plotting
plot_range = slice(25, 250)
plot_range1 = slice(5, 450)
plot_range2 = slice(15, 135)
