from pathlib import Path
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
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
c_thz = c0 * 1e-6  # µm / ps -> 1e6 / 1e-12 = 1e-6
eps0_thz = eps0 * 1e6 # F / m = S / (Hz * m) = Siemens * s / m = 1e12 ps * S / m = 1e6 ps * S / µm
