import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


# Load the Excel file with multiple sheets
excel_file = pd.ExcelFile('results.xlsx')

# Initialize an empty list to store the NumPy arrays
data_arrays = OrderedDict()

attack_indices = {'scale': range(0, 3),
                  'scale_minus': range(3, 6),
                  'noise': range(6, 9),
                  'flip_label': range(9, 12)}

check_indices = {'no_check': 0,
             'strict': 1,
             'prob_zkp': 2}

sheet_names = excel_file.sheet_names

# Process each sheet
for sheet_name in sheet_names:
    # Load the sheet into a DataFrame
    df = excel_file.parse(sheet_name, header=None)

    # Remove the first 4 rows
    df = df.iloc[4:]

    # Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()

    # Append the NumPy array to the list
    for attack, attack_index in attack_indices.items():
        data_arrays[(sheet_name, attack)] = data_array[:, attack_index]


def plot(sheet_name, attack, accuracies, ymin, ymax, legend):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 24

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    length = accuracies.shape[0]

    plt.plot(range(1, length + 1), accuracies[:length, check_indices['no_check']], linewidth=3)
    plt.plot(range(1, length + 1), accuracies[:length, check_indices['strict']], linewidth=3, alpha=0.6, linestyle='dashed')
    plt.plot(range(1, length + 1), accuracies[:length, check_indices['prob_zkp']], linewidth=3, linestyle='dotted')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    if legend:
        plt.legend(['RiseFL', 'NP-SC', 'NP-NC'])

    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f'{sheet_name}-{attack}.pdf', dpi=900, pad_inches=0, bbox_inches='tight')
    plt.clf()

def get_params(sheet_name, attack):
    ymin=0.5
    ymax=1
    return {'ymin': ymin, 'ymax': ymax, 'legend': False}

for sheet_name in sheet_names:
    for attack, attack_index in attack_indices.items():
        plot(sheet_name, attack,
             data_arrays[(sheet_name, attack)],
             **get_params(sheet_name, attack))
