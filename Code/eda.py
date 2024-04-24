import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def quantitative_analysis(df, target, excluded=None):
    num_columns =df.select_dtypes(['float' ,'int' ,'int64' ,'float32' ,'float64'])
    no_binary = num_columns.loc[:, ~(df.isin([0, 1 ,np.nan]).all())]
    if excluded!=None:
        for column in excluded:
            if column in no_binary.columns:
                no_binary.drop([column], axis=1 ,inplace=True)
    df1 =df.copy()
    df1.fillna(0, inplace=True)

    for column in no_binary.columns:
        # divide B into 10 bins with equal number of rows
        df_ok =df1[df1[column] != 0]
        # divide B into 15 bins with unique bin edges
        n_bins =15
        n_unique = df_ok[column].nunique()
        if n_unique < n_bins:
            n_bins = n_unique
        try:
            bins = pd.qcut(df_ok[column], q=n_bins, labels=False ,duplicates='drop')
            bin_values = [df_ok[column][bins == i].agg(['min', 'max']) for i in range(n_bins)]
            bin_labels = [f"{bin_values[i]['min']:.4f} - {bin_values[i]['max']:.4f}" for i in range(n_bins)]
            bin_labels =[item for item in bin_labels if item != "nan - nan"]
            df_ok['B_bin'] = pd.qcut(df_ok[column], q=n_bins, labels=bin_labels ,duplicates='drop')

            # compute the percentage of 1's and 0's in each bin
            grouped_1 = df_ok.loc[df1[target] == 1].groupby('B_bin').size().div(len(df_ok.loc[df1[target]==1]))
            grouped_0 = df_ok.loc[df1[target] == 0].groupby('B_bin').size().div(len(df_ok.loc[df1[target]==0]))

            # plot histograms
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(grouped_1.index, grouped_1, color='red', width=0.6, label=f'{target}')
            ax.bar(grouped_0.index, grouped_0, color='blue', width=0.4, label=f'no {target}')
            ax.set_xlabel(column)
            ax.set_ylabel('Percentage')
            ax.set_title(f"{target} vs {column}", fontsize=40)
            ax.legend()
            plt.xticks(rotation=90)
            plt.show()
        except ValueError:
            print(f"qcut failed for column '{column}'")

    corr_matrix =no_binary.corr()
    print(corr_matrix)

