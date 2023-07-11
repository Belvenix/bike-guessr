from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette(sns.color_palette("pastel"))

vis_folder = Path('visualize_training/data')


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


def visualize_epochs(df: pd.DataFrame, ax: plt.Axes, title: str):
    sns.lineplot(ax=ax, data=df, x='Step', y='Value', hue='model')
    ax.set_title(title)

    # Add X mark on the end of each line plot for each model
    unique_models = df['model'].unique()
    for model in unique_models:
        model_data = df[df['model'] == model]
        x_data = model_data['Step'].iloc[-1]
        y_data = model_data['Value'].iloc[-1]
        ax.plot(x_data, y_data, 'rx')


def main():
    train_connectedness_df = read_data(vis_folder / 'train_connectedness')
    val_connectedness_df = read_data(vis_folder / 'val_connectedness')
    train_f1_df = read_data(vis_folder / 'train_f1')
    val_f1_df = read_data(vis_folder / 'val_f1')

    # Create a figure with four subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    visualize_epochs(train_connectedness_df, axes[0][0], 'Train connectedness')
    visualize_epochs(val_connectedness_df, axes[0][1], 'Validation connectedness')
    visualize_epochs(train_f1_df, axes[1][0], 'Train F1')
    visualize_epochs(val_f1_df, axes[1][1], 'Validation F1')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Add main title
    # fig.suptitle('Metrics collected during training', fontsize=16)

    # Save the figure
    fig.savefig('visualize_training/visualize_training_metrics_alt.png')

    # Display the figure
    plt.show()


def main2():
    train_f1_df = read_data(vis_folder / 'old_train_f1')
    val_f1_df = read_data(vis_folder / 'old_val_f1')

    # Create a figure with four subplots
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    visualize_epochs(train_f1_df, axes[0], 'Train F1')
    visualize_epochs(val_f1_df, axes[1], 'Validation F1')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Add main title
    # fig.suptitle('Metrics collected during training', fontsize=16)

    # Save the figure
    fig.savefig('visualize_training/visualize_training_metrics_alt2.png')

    # Display the figure
    plt.show()


if __name__ == "__main__":
    main2()
