# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:28:04 2023

@author: MSI Mediamart
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path,encoding='latin-1')
cancer_data = load_data(path = "E:/cancer/Breast_Cancer_test.csv")
cancer_data.head()
cancer_data.dtypes
sns.heatmap(cancer_data.isnull(), yticklabels=False, cbar=False, cmap='cubehelix')

report = ProfileReport(cancer_data, title="Cancer data pandas profile report")
report.to_notebook_iframe()
num_patients: int = cancer_data.shape[0]
number_of_survivors: int = cancer_data["Status"].value_counts()["Alive"]

print(f"Of {num_patients} patients, {100 * (number_of_survivors / num_patients) :.3f}% survived")

num_white_people = cancer_data["Race"].value_counts()["White"]
num_black_people = cancer_data["Race"].value_counts()["Black"]
num_other_people = cancer_data["Race"].value_counts()["Other"]
#gán nhãn dữ liệu
white_survivors = cancer_data.loc[cancer_data["Status"] == "Alive"][cancer_data["Race"] == "White"]
black_survivors = cancer_data.loc[cancer_data["Status"] == "Alive"][cancer_data["Race"] == "Black"]
other_survivors = cancer_data.loc[cancer_data["Status"] == "Alive"][cancer_data["Race"] == "Other"]

num_white_survivors = white_survivors["Status"].value_counts()["Alive"]
num_black_survivors = black_survivors["Status"].value_counts()["Alive"]
num_other_survivors = other_survivors["Status"].value_counts()["Alive"]

print(f"Whites: Total: {num_white_people} - Survived: {num_white_survivors} - Percentage: {100 * (num_white_survivors / num_white_people) :.3f}%")
print(f"Blacks: Total: {num_black_people} - Survived: {num_black_survivors} - Percentage: {100 * (num_black_survivors / num_black_people) :.3f}%")
print(f"Others: Total: {num_other_people} - Survived: {num_other_survivors} - Percentage: {100 * (num_other_survivors / num_other_people) :.3f}%")


sns.countplot(data = cancer_data, x="Race", hue="Status", palette="husl")

stats = {}

for _, status in enumerate(list(cancer_data["Marital Status"].unique())):
    num_current_status = cancer_data["Marital Status"].value_counts()[status]
    status_survivors = cancer_data.loc[cancer_data["Status"] == "Alive"][cancer_data["Marital Status"] == status]
    num_survivors_status = status_survivors["Status"].value_counts()["Alive"]
    
    stats[status] = {"survivors": num_current_status, "num_people": num_survivors_status, "age": status_survivors["Age"].median()}

for key, value in stats.items():
    print(f"Total Number of {key} people: { value['survivors'] } have median age: {value['age'] :.3f}")
    print(f"{key}: Percentage survival - {100 * (value['num_people'] / value['survivors']) :.3f}%")
    print("=======================================================================================\n")
    
    

sns.countplot(data = cancer_data, x="Marital Status", hue="Status", palette="husl")

import itertools
marital_status = list(cancer_data["Marital Status"].unique())
race = list(cancer_data["Race"].unique())

for _, status in enumerate(itertools.product(marital_status, race)):
    current_df = cancer_data.loc[cancer_data["Marital Status"] == status[0]][cancer_data["Race"] == status[1]]
    num_patients: int = int(current_df["Marital Status"].value_counts())
    num_survivors: int = int(current_df["Status"].value_counts()["Alive"])
    
    
    print(f"Race: {status[1]} - Marital status: {status[0]}")
    print(f"Number of patients: {num_patients} with median age: {current_df['Age'].median() :.3f}")
    if num_survivors:
        print(f"Percentage survival: {100 * (num_survivors / num_patients) :.3f}%")
    else:
         print(f"Percentage survival: 0%")
    
    print("================================================================================================\n")
    
sns.catplot(data=cancer_data, x="Race", y="Age", kind="violin", color=".9", inner=None, hue="Status", split=True, palette="husl")

t_stage = list(cancer_data["T Stage "].unique())
n_stage = list(cancer_data["N Stage"].unique())
sixth_stage = list(cancer_data["6th Stage"].unique())
a_stage = list(cancer_data["A Stage"].unique())

stages = itertools.product(t_stage, n_stage, sixth_stage, a_stage)
for stage in stages:
    current_df = cancer_data.loc[cancer_data["T Stage "] == stage[0]][cancer_data["N Stage"] == stage[1]][cancer_data["6th Stage"] == stage[2]][cancer_data["A Stage"] == stage[3]]
    num_patients: int = current_df.shape[0]
    if not current_df.empty:
        num_survivors: int = current_df["Status"].value_counts()["Alive"]
        print(f"{num_patients} patients with Stage T: {stage[0]} | Stage N: {stage[1]} | Stage 6th: {stage[2]} | A stage: {stage[3]}")
        print(f"Percentage survival: {100 * (num_survivors / num_patients) :.3f}%")
        print("================================================================================================================================\n")
        
differentiation = list(cancer_data["differentiate"].unique())
grades = list(cancer_data["Grade"].unique())

for _, stage in enumerate(itertools.product(differentiation, grades)):
    current_df = cancer_data.loc[cancer_data["differentiate"] == stage[0]][cancer_data["Grade"] == stage[1]]
    if not current_df.empty:
        num_patients: int = current_df.shape[0]
        num_survivors: int = current_df["Status"].value_counts()["Alive"]
            
        print(f"{num_patients} patients with Differentiation Stage: {stage[0]} | Grade: {stage[1]}")
        print(f"Percentage survival: {100 * (num_survivors / num_patients) :.3f}%")
        print("================================================================================================================================\n")
        
progesterone = list(cancer_data["Progesterone Status"].unique())
oestrogen = list(cancer_data["Estrogen Status"].unique())

for _, status in enumerate(itertools.product(progesterone, oestrogen)):
    current_df = cancer_data.loc[cancer_data["Progesterone Status"] == status[0]][cancer_data["Estrogen Status"] == status[1]]
    if not current_df.empty:
        num_patients = current_df.shape[0]
        num_survivors: int = current_df["Status"].value_counts()["Alive"]
        
        progesterone_status = "Positive" if status[0] == 1 else "Negative"
        oestrogen_status = "Positive" if status[1] == 1 else "Negative"
        
        print(f"{num_patients} patients with  {status[0]} Progesterone and {status[1]} Oestrogen")
        print(f"Percentage survival: {100 * (num_survivors / num_patients) :.3f}%")
        print("================================================================================================================================\n")
        
cancer_data["Tumor Size"].describe()

sns.displot(cancer_data, x="Tumor Size", hue="Status", element="step", kde=True)

survived_cancer_data = cancer_data.loc[cancer_data["Status"] == "Alive"]["Tumor Size"]
dead_cancer_data = cancer_data.loc[cancer_data["Status"] == "Dead"]["Tumor Size"]

print(f"Patients who survived had tumor sizes of {survived_cancer_data.mean() :.3f} +/- {survived_cancer_data.std() :.3f} and median: {survived_cancer_data.median() :.3f}")
print(f"Patients who died had tumor sizes of {dead_cancer_data.mean() :.3f} +/- {dead_cancer_data.std() :.3f} and median: {dead_cancer_data.median() :.3f}")
from scipy.stats import ttest_ind

results = ttest_ind(survived_cancer_data, dead_cancer_data)

if results.pvalue < 0.001:
    print(f"{results.pvalue}: Results were highly significant ***")
elif results.pvalue < 0.01:
    print(f"{results.pvalue}: Results were moderately significant **")
elif results.pvalue < 0.05:
    print(f"{results.pvalue}: Results were significant *")

cancer_sample = cancer_data[["Regional Node Examined", "Reginol Node Positive", "Status"]]
cancer_sample["Percentage Positive Nodes"] = 100 * (cancer_sample["Reginol Node Positive"] / cancer_sample["Regional Node Examined"])

cancer_sample.describe()

nodes_survived = cancer_sample.loc[cancer_sample["Status"] == "Alive"]["Percentage Positive Nodes"]
nodes_dead = cancer_sample.loc[cancer_sample["Status"] == "Dead"]["Percentage Positive Nodes"]

print(f"Patients who survived had {nodes_survived.mean() :.3f}% +/- {nodes_survived.std() :.3f} positive nodes with median: {nodes_survived.median() :.3f}%")
print(f"Patients who died had {nodes_dead.mean() :.3f}% +/- {nodes_dead.std() :.3f} positve nodes with median: {nodes_dead.median() :.3f}%")

results = ttest_ind(nodes_survived, nodes_dead)

if results.pvalue < 0.001:
    print(f"{results.pvalue}: Results were highly significant ***")
elif results.pvalue < 0.01:
    print(f"{results.pvalue}: Results were moderately significant **")
elif results.pvalue < 0.05:
    print(f"{results.pvalue}: Results were significant *")