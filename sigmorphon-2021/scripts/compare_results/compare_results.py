import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


# gets the names of the result files as parameters and exports a jpg in the same folder
def main():
    x_axis_name = "base on low languages"
    y_axis_name = "augmentation strategies on low languages"
    first_data_name = sys.argv[1]
    second_data_name = sys.argv[2]
    first = pd.read_csv(first_data_name, names=['language', x_axis_name],
                        header=None)
    second = pd.read_csv(second_data_name, names=['language', y_axis_name])

    merged = pd.merge(first, second, on=['language'], how='inner')

    sns.scatterplot(data=merged, x=x_axis_name, y=y_axis_name)
    plt.plot([0, 1], [0, 1], linewidth=.5, color='red')

    for index, row in merged.iterrows():
        plt.text(row[x_axis_name]-.03, row[y_axis_name]+.03, row['language'])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(first_data_name + "_" + second_data_name + "_" + "_scatter.jpg")


if __name__ == "__main__":
    main()
