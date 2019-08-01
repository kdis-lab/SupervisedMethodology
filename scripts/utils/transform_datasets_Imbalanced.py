import pandas as pd

fileName= "CHOL"

data = pd.read_csv("./datasets/{0}.txt".format(fileName), delimiter="\t")

t_data = data.transpose()

indexes = t_data.index.values

classes = []

for index in indexes:


    if fileName != "SKCM":
    
        if ("-01A" in index) or ("-01B" in index) or ("-06A" in index) or ("-02B" in index) or ("-02A" in index) or ("-01C" in index):

            classes.append("T")

        if "-11A" in index:

            classes.append("N")
    
    else:

        if ("-01A" in index) or ("-01B" in index):

            classes.append("T")

       # then remove 11a manually
        if ("-06A" in index) or ("-06B" in index) or ("-07A" in index):

            classes.append("N")

if fileName != "SKCM":
    if len(classes) != data.shape[1]:

        print("Something goes wrong", len(classes), data.shape[1])
        exit(1)

# Appending class column
t_data["Class"] = classes

# replacing column names by adding the corresponding gene names

columns = t_data.columns.values

ig = pd.read_csv("datasets/iso-gen.txt", delimiter="\t")

columns_list =[]

gen_isoform = {}

for row in ig.iterrows():

    gen_isoform[row[1]["Isoform"]]= row[1]["Gen"]

for column in columns:

    gen = gen_isoform.get(column, "")

    if gen == "":

        columns_list.append(column)
    
    else:
        
        columns_list.append(gen + "," + column)

t_data.columns = columns_list

t_data.to_csv("datasets/{0}-preproc.csv".format(fileName), sep=",")

