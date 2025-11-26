from dataset import DataSet

options = {}

climate_log_file = r"/home/ftpuser/ftp/Data/Stability/2023-03-21 12-27-00_log.txt"
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Stability/terak15/20avg_6_lab1_woodenlenssetup (copy)", options)
dataset = DataSet(r"/home/ftpuser/ftp/Data/Stability/2023-03-21", options)

dataset.select_freq(freq=1.2)

dataset.plot_system_stability(climate_log_file)
dataset.plt_show()
