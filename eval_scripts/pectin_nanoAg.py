from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os
from pathlib import Path

options = {
"ref_pos": (20, None),

"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",

"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.05, # 0.999, # 0.99
                          # "win_start": 0,
                          "win_width": 70, # 18,#2*32,# 38*2, # 5*15 # 36
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },
"eval_opt": {"fit_range": (0.50, 2.20),},
"plot_opt" : {"plot_range": (0.25, 3.9)},
"shown_plots": {
    "Window": False,
    "Time domain": False,
    "Spectrum": True,
    "Phase": False,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": True,
    "Absorption coefficient": True,
    "Conductivity": False,
},
}
"""
HE_A1: 100, 0
HE_A2: 74, 0
HE_C: 90, 0
LE_A: 80, 0
LE_AP: 90, 0
LE_C: 90, 0
"""

"""
meas_dict = {# "HE_A1": (100, 0), #"HE_A2": (74, 0), "HE_C": (90, 0),
             "LE_A": (80, 0), "LE_A-90": (90, 0), "LE_A-95": (95, 0),
            #"LE_AP": (90, 0), "LE_C": (90, 0),
}

for meas_set in meas_dict:
    if "-" in meas_set:
        dir_name = meas_set.split("-")[0]
    else:
        dir_name = meas_set

    if 'nt' in os.name:
        sam_dataset_path = fr"C:\\Users\\alexj\Data\Pectin_wAg_Nanoparticles\{dir_name}"
    else:
        sam_dataset_path = fr"/home/ftpuser/ftp/Data/Pectin_wAg_Nanoparticles/{dir_name}"

    dataset = DataSet(sam_dataset_path, options)
    pos = meas_dict[meas_set]
    dataset.plot_meas(pos, label=meas_set, err_bar_limits=(70, 95))
"""
meas_points = {"Sample7_1": (84, -18), "Sample7_2": (83, -18), "Sample7_3": (83, -18),
               "Sample6_1": (84, 0), "Sample6_2": (83, 0), "Sample6_3": (82, 0),
               "Sample5_1": (67, 0), "Sample5_2": (66, 0), "Sample5_3": (65, 0),
               "Sample4_1": (49, 0), "Sample4_2": (48, 0), "Sample4_3": (47, 0),
               "Sample3_1": (49, -18), "Sample3_2": (48, -18), "Sample3_3": (47, -18),
               "Sample2_1": (67, -18), "Sample2_2": (66, -18), "Sample2_3": (65, -18),
               # Sample 1 is missing (measured sample 3 twice because of wrong coordinates)
}
sample_thicknesses = {"Sample1": 89, "Sample2": 98, "Sample3": 92, "Sample4": 98,
                      "Sample5": 101, "Sample6": 112, "Sample7": 111}
sample_thicknesses = {"Sample1": 89, "Sample2": 98, "Sample3": 92, "Sample4": 98,
                      "Sample5": 101, "Sample6": 112, "Sample7": 111}

if 'nt' in os.name:
    base_dir = fr""
else:
    base_dir = Path(fr"/home/ftpuser/ftp/Data/Pectin_set2")

for spot in range(4):
    for meas_point in meas_points:
        if spot != 2:
            continue
        if str(spot) != meas_point[-1]:
            continue
        if "Sample6" not in meas_point:
            continue

        if "Sample7" in meas_point:
            dir_name = base_dir / "2026.04.24" / "Measurement_set_sample_7"
        else:
            dir_name = base_dir / "2026.04.21" / "Measurement_Set_1"

        options["sample_properties"] = {"d": sample_thicknesses[meas_point.split("_")[0]]}

        dataset = DataSet(dir_name, options)
        pos = meas_points[meas_point]

        dataset.plot_meas(pos, label=meas_point, fig_num_ext=f"_{spot}")


dataset.plt_show()
