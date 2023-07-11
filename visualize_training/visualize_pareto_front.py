import typing as tp
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette(sns.color_palette("pastel"))

VIS_FOLDER = Path('visualize_training/data')
MODEL_NAMES = ['Trivial', 'MGCN', 'GNN', 'MGCN no encoding', 'GNN no encoding']


def read_data(folder: Path):
    mgcn = pd.read_csv(folder / 'mgcn.csv')
    mgcn['model'] = 'MGCN'
    gnn = pd.read_csv(folder / 'gnn.csv')
    gnn['model'] = 'GCN'
    trivial = pd.read_csv(folder / 'trivial.csv')
    trivial['model'] = 'Trivial'
    mgcn_no_encoding = pd.read_csv(folder / 'mgcn-no-encoding.csv')
    mgcn_no_encoding['model'] = 'MGCN no encoding'
    gnn_no_encoding = pd.read_csv(folder / 'gnn-no-encoding.csv')
    gnn_no_encoding['model'] = 'GCN no encoding'
    return pd.concat([mgcn, gnn, trivial, mgcn_no_encoding, gnn_no_encoding], ignore_index=True)


def calculate_pareto_frontier(data):
    pareto_frontier = [data[0]]  # Initialize the Pareto frontier with the first data point

    for point in data[1:]:
        # Check if the current point dominates any point on the Pareto frontier
        is_dominated = False
        frontier_copy = pareto_frontier.copy()  # Create a copy to avoid modifying the frontier during iteration

        for frontier_point in frontier_copy:
            if point['f1'] <= frontier_point['f1'] and point['connectedness'] <= frontier_point['connectedness']:
                # If the current point is worse or equal in both metrics, it is dominated
                is_dominated = True
                break

            if point['f1'] >= frontier_point['f1'] and point['connectedness'] >= frontier_point['connectedness']:
                # If the current point is better or equal in both metrics, it dominates the frontier point
                pareto_frontier.remove(frontier_point)

        if not is_dominated:
            pareto_frontier.append(point)  # Add the current point to the Pareto frontier

    return pareto_frontier


def plot_pareto_frontier(d: tp.List[dict], pf: tp.List[dict], fig, ax, name, pareto_color, draw_bar=False):
    """Plot pareto frontier and consecutive epochs line.

    Args:
        d (tp.List[dict]): All data points to plot.
        pf (tp.List[dict]): Pareto frontier data points to plot.
    """
    # Extract metric values from Pareto frontier
    pf_copy = pf.copy()
    pf_copy.sort(key=lambda x: x['f1'])
    pareto_A = [point['f1'] for point in pf_copy]
    pareto_B = [point['connectedness'] for point in pf_copy]

    d.sort(key=lambda x: x['epoch'])

    # Create a continuous pastel gradient color map
    cmap = mcolors.LinearSegmentedColormap.from_list('pastel_gradient', ['#FFD700', '#FF7F50'])

    # Extract data points
    x = [point['f1'] for point in d]
    y = [point['connectedness'] for point in d]
    c = [point['epoch'] for point in d]

    # Plotting the Pareto frontier
    ax1 = ax.scatter(x, y, c=c, marker='x', cmap=cmap)
    if draw_bar:
        fig.colorbar(ax1, label='Epoch')

    ax.plot(pareto_A, pareto_B, marker='o', color=pareto_color, label=name)


def retrieve_pareto_data(df_c: pd.DataFrame, df_f: pd.DataFrame):
    all_models = [
        {
            'f1': f['Value'],
            'connectedness': c['Value'],
            'epoch': f['Step'],
            'model': f['model']
        }
        for c, f in zip(df_c, df_f)
    ]
    return [list(filter(lambda x: x['model'] == m, all_models)) for m in MODEL_NAMES]


def main():
    # read data
    train_connectedness_df = read_data(VIS_FOLDER / 'train_connectedness')
    tc = train_connectedness_df.to_dict('records')

    val_connectedness_df = read_data(VIS_FOLDER / 'val_connectedness')
    vc = val_connectedness_df.to_dict('records')

    train_f1_df = read_data(VIS_FOLDER / 'train_f1')
    tf = train_f1_df.to_dict('records')

    val_f1_df = read_data(VIS_FOLDER / 'val_f1')
    vf = val_f1_df.to_dict('records')

    # transform data
    pareto_train_data = retrieve_pareto_data(tc, tf)
    pareto_val_data = retrieve_pareto_data(vc, vf)

    fig, axes = plt.subplots(ncols=2, nrows=3, sharey='all', figsize=(12, 8))
    fig.suptitle('Pareto front in consecutive epochs for: ')

    # plot pareto frontier train
    for ptd, pvd, model_name, ax in zip(pareto_train_data, pareto_val_data, MODEL_NAMES, axes.flatten()):
        # Calculate pareto frontier for train and val data
        pf_t, pf_v = calculate_pareto_frontier(ptd), calculate_pareto_frontier(pvd)

        plot_pareto_frontier(ptd, pf_t, fig, ax, 'train', '#F49AC2', draw_bar=True)
        plot_pareto_frontier(pvd, pf_v, fig, ax, 'val', '#B19CD9')

        ax.axhline(y=1, color='#AEC6CF', linestyle='--', linewidth=2, label='Max(C)')
        ax.axvline(x=1, color='#B7D1A1', linestyle='--', linewidth=2, label='Max(F1)')
        ax.set_xlabel('Metric F1')
        ax.set_ylabel('Metric Connectedness')
        # set limits <0,1.5> for both axes

        # ax.set_xlim(0, 1.25)
        # ax.set_ylim(0, 1.25)
        ax.set_title(model_name)
        ax.set_yscale('log')
        ax.legend()

    # Adjust spacing between subplots
    fig.delaxes(axes[2, 1])
    plt.subplots_adjust(hspace=0.5)
    # Update the layout
    fig.tight_layout()
    fig.savefig(VIS_FOLDER / 'all_pareto_frontiers.png')
    fig.show()


if __name__ == '__main__':
    main()
    print('Done')
