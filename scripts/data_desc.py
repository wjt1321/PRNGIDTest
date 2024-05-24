# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:02:26 2024

@author: jarre
"""

import pandas as pd
import numpy as np
import seaborn as sns
import glob

df = pd.read_csv("moutput.csv")
df = df.drop(["seed"],axis="columns")


def pass_fail(alpha,df):
    unique_gens = df["gtype"].unique()
    dfmod = df.drop(["parameter"],axis="columns")
    percentages_df = pd.DataFrame(data=None, columns=dfmod.columns)
    sub_lengths =[]
    num_params=[]
    for label in unique_gens:
        subdf = df[df["gtype"]==label]
        #print()
        #pl = subdf["parameter"].unqiue()
        #num_params.append(len(pl))
        subdf = subdf.drop(["gtype","parameter"],axis="columns")
        number_of_rows = len(subdf)
        sub_lengths.append(number_of_rows)
        #clm = list(subdf)
        #ls2=[]
        #, index=subdf.index)
        #print(df2)
        ls = [label]
        for column in subdf.columns:
            count_within_range1 = subdf[subdf[column].between(0, alpha)].shape[0]
            count_within_range2 = subdf[subdf[column].between(1-alpha, 1)].shape[0]
            total_values = subdf[column].shape[0]
            percentage_within_range1 = (count_within_range1 / total_values) * 100
            percentage_within_range2 = (count_within_range2 / total_values) * 100
            precentage_pass = 100-(percentage_within_range1 + percentage_within_range2)
            ls.append(precentage_pass)
        percentages_df.loc[len(percentages_df)] = ls
    return percentages_df,sub_lengths,num_params

def main():
    df = pd.read_csv("moutput.csv")
    df = df.drop(["seed"],axis="columns")
    try:
        alpha = eval(input("Please enter an alpha value between 0 and 1: "))
        if (alpha <=0) or (alpha >=1):
            print("invalid alpha, using alpha=0.05")
            alpha = 0.05
    except: 
        print("invalid alpha, using alpha=0.05")
        alpha = 0.05
    percents_df,sub_lengths,num_params = pass_fail(alpha, df)
    pass_rate = percents_df.drop("gtype",axis="columns").sum(axis=1)/(106*100)
    Percent_of_whole = np.around((np.array(sub_lengths)/len(df))*100,1)
    output_df = pd.DataFrame(data=None,columns = ["gtype","avg_Pass_Rate","total_Number","percent_Of_Whole"])
    labs = percents_df["gtype"]
    output_df["gtype"] = labs
    output_df["avg_Pass_Rate"] = np.around(pass_rate*100,1)
    output_df["total_Number"] = sub_lengths
    output_df["percent_Of_Whole"] = Percent_of_whole
    #output_df["params_tested"] = num_params
    print(f"for alpha {alpha}:\n")
    print(output_df)
    print("_____________________________________________________________________________\n")
    print(percents_df)
    return(percents_df)
    output_df.to_csv('C:/Users/jarre/Downloads/Results/table.csv',index=False)
# print(ls1)

if __name__ =="__main__":
    percents = main()
    