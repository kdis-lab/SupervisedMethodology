import pandas as pd

fileName= "thca"

data = pd.read_csv("./datasets/{0}.txt".format(fileName), delimiter="\t")

t_data = data.transpose()

indexes = t_data.index.values

classes = []

for index in indexes:
    
    if index[-1] == 'T':

        classes.append("T")

    if index[-1] == 'N':

        classes.append("N")

if len(classes) != data.shape[1]:

    print("Something goes wrong", len(classes), data.shape[1])
    exit(1)

# Appending class column
t_data["Class"] = classes

t_data.to_csv("datasets/{0}-preproc.csv".format(fileName), sep=",")