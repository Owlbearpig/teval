import numpy as np
from consts import os_name, Path, result_dir
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib as mpl


def mpl_style_params(new_rcparams=None):
    rcParams = mpl.rcParams
    # rcParams['lines.linestyle'] = '--'
    # rcParams['legend.fontsize'] = 'large' #'x-large'
    rcParams['legend.shadow'] = False
    # rcParams['lines.marker'] = 'o'
    rcParams['lines.markersize'] = 6
    rcParams['lines.linewidth'] = 24 # 3.5
    rcParams['ytick.major.width'] = 2.5
    rcParams['xtick.major.width'] = 2.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.grid'] = False # True
    rcParams['figure.autolayout'] = False
    rcParams['savefig.format'] = 'png'
    rcParams["scatter.marker"] = "o"  # "x"
    rcParams.update({'font.size': 32})

    # Say, "the default sans-serif font is COMIC SANS"
    # rcParams['font.sans-serif'] = 'Liberation Sans'
    # Then, "ALWAYS use sans-serif fonts"
    # rcParams['font.family'] = "sans-serif"
    rcParams.update({
        "text.usetex": False,  # Use LaTeX to write all text
        # "pgf.texsystem": "pdflatex",
        "font.family": "serif",  # Use serif fonts
        # "font.serif": ["Computer Modern"],  # Ensure it matches LaTeX default font
        # "text.latex.preamble": r"\\usepackage{amsmath}"  # Add more packages as needed
        #"pgf.preamble": "\n".join([ # plots will use this preamble
        #r"\usepackage[utf8]{inputenc}",
        #r"\usepackage[T1]{fontenc}",
        #r"\usepackage{siunitx}",
        #])
    })
    rcParams["savefig.directory"] = result_dir

    if new_rcparams:
        rcParams.update(new_rcparams)

    Path(rcParams["savefig.directory"]).mkdir(parents=True, exist_ok=True)

    return rcParams


def test_plot():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    dt = 0.01
    t = np.arange(0, 10, dt)
    nse = np.random.randn(len(t))
    r = np.exp(-t / 0.05)

    cnse = np.convolve(nse, r) * dt
    cnse = cnse[:len(t)]
    s = 0.1 * np.sin(2 * np.pi * t) + cnse

    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.plot(t, s)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Signal')
    ax1.psd(s, 512, 1 / dt)


if __name__ == '__main__':
    import logging
    from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show
    import matplotlib as mpl

    print([f.name for f in matplotlib.font_manager.fontManager.ttflist], "\n")
    print(mpl.rcParams.keys())

    logging.basicConfig(level=logging.INFO)
    mpl.rcParams = mpl_style_params()

    test_plot()

    fig, ay = subplots()

    # Using the specialized math font elsewhere, plus a different font
    xlabel(r"The quick brown fox jumps over the lazy dog", fontsize=18)
    # No math formatting, for comparison
    ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
    grid()

    show()

