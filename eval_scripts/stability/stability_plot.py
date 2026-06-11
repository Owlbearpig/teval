from dataset import DataSet

climate_log_file = r"/home/ftpuser/ftp/Data/Stability/2023-03-21 12-27-00_log.txt"
# climate_log_file = r"/media/storage/ArchivedData/Stability/T_RH_sensor_logs/2024-11-16 11-27-45_log.txt"


# dataset = DataSet(r"/home/ftpuser/ftp/Data/Stability/terak15/20avg_6_lab1_woodenlenssetup (copy)")
dataset = DataSet(r"/home/ftpuser/ftp/Data/Stability/2023-03-21")
# dataset = DataSet(r"/media/storage/ArchivedData/Stability/100 ps 20 avg")

dataset.select_freq(freq=1.2)

dataset.plot_system_stability()
# dataset.plot_climate(climate_log_file)
dataset.plt_show()
