import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quantitative_analysis(df, target, excluded=None):
    # Select numerical columns
    num_columns = df.select_dtypes(['float', 'int', 'int64', 'float32', 'float64'])
    # Exclude binary columns
    no_binary = num_columns.loc[:, ~(df.isin([0, 1, np.nan]).all())]

    # Exclude specified columns
    if excluded is not None:
        for column in excluded:
            if column in no_binary.columns:
                no_binary.drop([column], axis=1, inplace=True)

    # Make a copy of the dataframe and fill NaN values with 0
    df1 = df.copy()
    df1.fillna(0, inplace=True)

    # Iterate over non-binary columns
    for column in no_binary.columns:
        # Filter out rows where the value of the current column is 0
        df_ok = df1[df1[column] != 0]

        # Determine the number of bins to use for binning
        n_bins = 15
        n_unique = df_ok[column].nunique()
        if n_unique < n_bins:
            n_bins = n_unique

        try:
            # Perform quantile binning
            bins = pd.qcut(df_ok[column], q=n_bins, labels=False, duplicates='drop')
            bin_values = [df_ok[column][bins == i].agg(['min', 'max']) for i in range(n_bins)]
            bin_labels = [f"{bin_values[i]['min']:.4f} - {bin_values[i]['max']:.4f}" for i in range(n_bins)]
            bin_labels = [item for item in bin_labels if item != "nan - nan"]
            df_ok['B_bin'] = pd.qcut(df_ok[column], q=n_bins, labels=bin_labels, duplicates='drop')

            # Compute the percentage of 1's and 0's in each bin
            grouped_1 = df_ok.loc[df1[target] == 1].groupby('B_bin').size().div(len(df_ok.loc[df1[target] == 1]))
            grouped_0 = df_ok.loc[df1[target] == 0].groupby('B_bin').size().div(len(df_ok.loc[df1[target] == 0]))

            # Plot histograms
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

    # Compute correlation matrix
    corr_matrix = no_binary.corr()
    print(corr_matrix)