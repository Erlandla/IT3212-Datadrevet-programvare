import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt 

def read_results(directory:str = 'result'):
    results = {}
    for folder in os.listdir(directory):
        folder_name = os.path.join(directory, folder)
        for inner_folder in os.listdir(folder_name):
            inner_folder_name = os.path.join(folder_name, inner_folder)
            print(inner_folder_name)

            for json_file in os.listdir(inner_folder_name):
                json_file_name = os.path.join(inner_folder_name, json_file)
                if json_file == 'test_result.json':
                    test_results = json.load(open(json_file_name))
                    print(test_results['rmae'])
                    results[inner_folder_name] = test_results['rmae']
    return results

def plot_results(results):
    labels = list(results.keys())
    rmaes = list(results.values())

    fig = plt.figure(figsize = (10, 5))
    plt.bar(labels, rmaes, color ='maroon', width = 0.4)
    fig.subplots_adjust(bottom=0.6)
    plt.xticks(rotation=90)

    plt.xlabel("Tuning states")
    plt.ylabel("Score/error")
    plt.title("Hyperparameter tuning")
    plt.show()
    

results = read_results()
plot_results(results)