import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import B_FILTER, L_FILTER, M_FILTER, R_FILTER, S_FILTER, T_FILTER, VISUALIZATION_OUTPUT_DIR

search_path = VISUALIZATION_OUTPUT_DIR.glob('*_recent_edge_counts.csv')


df = pd.DataFrame()
dfs = []
for file in search_path:
    dfs.append(pd.read_csv(file))

merged_df = pd.DataFrame({'name': []})
merged_df = pd.concat(dfs, ignore_index=True).groupby('name').sum().reset_index()
# combine filter names and their respective counts
print(merged_df)


def set_filter(row):
    row_name = row['name']
    if row_name in L_FILTER:
        return 'L'
    if row_name in M_FILTER:
        return 'M'
    if row_name in B_FILTER:
        return 'B'
    if row_name in T_FILTER:
        return 'T'
    if row_name in S_FILTER:
        return 'S'
    if row_name in R_FILTER:
        return 'R'
    if row_name == 'nocycle':
        return 'nocycle'
    else:
        raise ValueError(f'Filter {row_name} not found')


def set_class(row):
    filter_name = row['filter']
    if filter_name in ['L', 'M', 'B']:
        return 'lane'
    if filter_name in ['T', 'S']:
        return 'track'
    if filter_name in ['R']:
        return 'road'
    if filter_name == 'nocycle':
        return 'nocycle'
    else:
        raise ValueError(f'Filter {filter_name} not found')


merged_df['filter'] = merged_df.apply(set_filter, axis=1)
merged_df['class'] = merged_df.apply(set_class, axis=1)
merged_df = merged_df.groupby(['filter', 'class']).sum().reset_index()

sns.set_palette(sns.color_palette("pastel"))
sns.barplot(x='filter', y='count', hue='class', data=merged_df, dodge=False)
plt.yscale('log')
plt.xlabel('Filter')
plt.ylabel('Count')
plt.title('Downloaded Filter Counts for train, validation, and test sets')
plt.savefig('downloaded_filter_counts.png')
plt.show()
print(merged_df)
