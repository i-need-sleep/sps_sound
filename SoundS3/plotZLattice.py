from typing import List, Dict

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from numpy import iterable
from tqdm import tqdm

FIGSIZE = (8, 4)
# WIDTH_RATIO = (.07, .9)

import rc_params
rc_params.init()
from linearity_shared import *

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'serif', 
    'font.serif': ['Computer Modern'], 
    'font.size': 14, 
})

def main():
    fig = plt.figure(
        constrained_layout=True, figsize=FIGSIZE, 
    )
    # subfigs: List[SubFigure] = fig.subfigures(1, 2, width_ratios=WIDTH_RATIO)
    subfigs: List[SubFigure] = [fig.subfigures(1)]
    # subfig_0_ax: Axes = subfigs[0].subplots()
    # subfig_0_ax.axis('off')
    axeses: List[List[Axes]] = subfigs[-1].subplots(
        2, 4, 
        # sharey=True, 
        sharex=False, 
    )
    # axeses = [[*axeses]]
    # axeses = [[*axeses[:, 0], *axeses[:, 1]]]
    plotted: Dict[str, List[Line2D]] = {}
    for task_i, (
        task_path_name, task_display, 
        (x_path, x_display), 
        (y_path, y_display), 
        plt_style, 
    ) in enumerate(TASKS):
        n_cols = len(EXP_GROUPS)
        if task_path_name == 'decode' and SPICE in [x[1] for x in EXP_GROUPS]:
            # n_cols -= 1 # for SPICE
            axeses[task_i][n_cols].axis('off')
        for col_i in tqdm(range(n_cols), task_display):
            exp_group = EXP_GROUPS[col_i]
            ax: Axes = axeses[task_i][col_i]
            is_spice = exp_group[1] == SPICE
            # extract X, Y
            result_path = RESULT_PATH % (task_path_name, 'test_set', exp_group[1])
            data = readXYFromDisk(
                is_spice, result_path, x_path, y_path,
            )
            # plot X, Y
            for instrument_name, (X, Y) in data.items():
                if instrument_name not in plotted:
                    plotted[instrument_name] = []
                plotted[instrument_name].append(ax.plot(
                    X, Y, label=instrument_name, **plt_style
                )[0])
            if task_i == 0:
                ax.set_title(exp_group[0])
            ax.set_xlabel(x_display)
            if col_i == 0:
                ax.set_ylabel(y_display)
                ax.annotate(
                    task_display, 
                    xy=(0, 0.5), 
                    xytext=(-ax.yaxis.labelpad - 10, 0), 
                    xycoords=ax.yaxis.label, 
                    textcoords='offset points', 
                    ha='right', va='center', 
                    size='large',
                    rotation=90, 
                )
            if task_path_name == 'decode':
                ax.set_yticks((60, 84))
                ax.set_yticklabels(('C4', 'C6'))
                ax.set_ylim((55, 89))
                ax.set_xlim((-1.1, 1.1))
            else:
                ax.set_xticks((60, 84))
                ax.set_xticklabels(('C4', 'C6'))
                ax.set_yticks((-1, 0, 1))
                ax.set_yticklabels(('-1', '0', '1'))
                ax.set_ylim((-1.3, 1.3))
    def prettify(k):
        return k.replace('Electric ', 'E. ')
    K, V = [], []
    for k in sorted(
        plotted.keys(), reverse=True, 
        key=lambda k : len(prettify(k))
    ):
        K.append(prettify(k))
        V.append(plotted[k][0])
    
    axeses[0][-1].legend(
        V, K, markerscale=2, 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1.05), 
        # fontsize=10, 
        labelspacing=1, 
        ncol=2,
        handlelength=0.3,
        handletextpad=0.5,
        columnspacing=.8,
    )
    plt.savefig(f'./figs/cool_fig.pdf')
    plt.show()

if __name__ == '__main__':
    main()
