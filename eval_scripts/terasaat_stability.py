from dataset import DataSet, plt_show

options = {}
dataset = DataSet(r"/home/ftpuser/ftp/Data/Stability/terak15/20avg_6_lab1_woodenlenssetup (copy)", options)

dataset.plot_system_stability()

plt_show()
