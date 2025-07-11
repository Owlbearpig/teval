from pathlib import Path
from scipy.constants import c as c0
from numpy import pi
import os
from os import name as os_name

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

project_ = "Conductivity"

if 'posix' in os_name:
    # result_dir = Path(r"/home/alex/MEGA/AG/Projects") / project_ / "IPHT_2" / "Results" / "Initial_characterization"
    result_dir = Path(r"/home/alex/MEGA/AG/Projects") / project_ / "Furtwangen" / "Results" / "Vanadium"
else:
    result_dir = Path(r"C:\Users\alexj\Mega\AG\Projects") / project_

# physical constants
THz = 1e12
c_thz = c0 * 1e-6  # um / ps -> 1e6 / 1e-12 = 1e-6

# plotting
plot_range1 = slice(15, 900)
plot_range2 = slice(15, 900)
