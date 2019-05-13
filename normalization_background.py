import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import csv 
classes = np.array(['./16Fa CHEM 1P', './16S1 PHY 3A',
           './17Fa PUBHLTH 1', './17Fa PUBHLTH 2', './17S1 PHY 3A',
           './17Sp PUBHLTH 1', './17Sp PUBHLTH 2', './17Wi PUBHLTH 1',
           './17Wi PUBHLTH 2', './18S1 CHEM 1C',
           './18Wi BIO SCI 9B'])

i = 0
overall_data = None
indices = []
for cla in classes:
    data = pd.read_csv('./training_data/' + cla[2:] + "_background_training.csv")
    data.head()
    if i == 0:
        overall_data = data.values.astype(float)
        i += 1
        indices.append(0)
    else:
        overall_data = np.concatenate((overall_data, data.values.astype(float)))

    indices.append(overall_data.shape[0])

print(indices)
for i in range(4):
    overall_data[:, i] = preprocessing.scale(overall_data[:, i].astype(float))


    
for i in range(len(classes)):
    lines = np.array( [["age", "sattotalscore", "hsgpa",
                      "gpacumulative", "istransfer", "ismale", "lowincomeflag",
                      "firstgenerationflag", "isurm", "grade", "GPA"]])
    lines = np.concatenate((lines, overall_data[indices[i]:indices[i+1], :]))
    print(lines.shape)

    
    file_name = "./training_data/" + classes[i][2:] + "_background_training.csv"
    with open(file_name, 'w', newline='') as f:

        writer = csv.writer(f, delimiter=',')
        writer.writerows(lines)
    
