from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os
import numpy as np

options = {

"plot_range": slice(13, 230),

"ref_pos": (0, None),
"fix_ref": 0,
"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"save_plots": True,
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.05, # 0.999, # 0.99
                          # "win_start": 0,
                          "win_width": 70, # 18,#2*32,# 38*2, # 5*15 # 36
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },

"shown_plots": {
    "Window": False,
    "Time domain": False,
    "Spectrum": False,
    "Phase": False,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": True,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": False,
},
}

samplesets = {"PA6": (65, 0), "PA_E73": (62, 0), "PBT_GF3D": (65,0), "PC": (68,0), "PDT_GF3D": (63,0),
              "PE_HD": (64,-10), "PMMA": (65,0), "PMMA_fine": (65,0), "PP": (64,0), "PP_H": (64,-10),
              "Sikaflex_521": (50,0), "Sikaflex_521_Metall_fine": (72,0),
              "Sikaflex_UHM": (51,0), "Sikaflex_UHM_Metall_fine": (72,0),
              }
limits = {"PA6": (63, 69), "PA_E73": (58, 67), "PBT_GF3D": (62,69), "PC": (66,70), "PDT_GF3D": (59,68),
          "PE_HD": (61.5,68), "PMMA": (62,88), "PMMA_fine": (71,73), "PP": (61,67), "PP_H": (61.5,67.5),
          "Sikaflex_521": (42,58), "Sikaflex_521_Metall_fine": (70,73),
          "Sikaflex_UHM": (42,57), "Sikaflex_UHM_Metall_fine": (71,77),}

rets = []
for sample_name, pos in samplesets.items():
    if 'nt' in os.name:
        sam_dataset_path = fr"C:\Users\alexj\Data\Gi_Machbarkeitsstudie_2\{sample_name}"
    else:
        sam_dataset_path = fr"/home/ftpuser/ftp/Data/Gi_Machbarkeitsstudie_2/{sample_name}"

    dataset = DataSet(sam_dataset_path, options)

    dataset.select_quantity(QuantityEnum.P2P)
    dataset.select_freq(1.00)
    # dataset.plot_line(line_coords=16, fig_num_=sample_name)
    label = sample_name.replace("_fine", "")
    ret = dataset.plot_point(pos,
                             label=label,
                             err_bar_limits=limits[sample_name],
                             ref_err_bars=True)

    rets.append((sample_name, 20*np.log10(ret["absorb"])[dataset._selected_freq_idx()]))
    # dataset.plt_show(save_file_suffix=sample_name, only_save_plots=True)
    dataset.plt_show()

for ret in rets:
    print(ret)