# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:32:53 2024

@author: jarre
"""

import pandas as pd
import numpy as np
from scipy.stats import t
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def calculate_confidence_intervals(data,confidence_level):
    degrees_of_freedom = len(data) - 1
    mean, std_dev = data.mean(), data.std()
    if degrees_of_freedom < 1 :
        print("degree error")
    
    confidence_interval = t.interval(confidence_level, degrees_of_freedom, loc=mean, scale=std_dev / (len(data) ** 0.5))
    return confidence_interval
def casetoprediction(case):
    out=""
    if case < 3:
       out = "CMRG" 
    elif case <10:
        out = "LCG"
    elif case <13:
        out = "LCG64"
    elif case<24:
        out = "LFG"
        #print(f"predicted LFG: {case}")
    elif case < 35:
        out = "MLFG"
        #print(f"predicted MLFG: {case}")
    elif case == 35:
        out = "PMLCG"
    return out

def predictions(ls):
    predicts = [0,0,0,0,0,0]
    for l in ls:
        if l < 3:
           predicts[0]+=1
        elif l <10:
            predicts[1]+=1
        elif l <13:
            predicts[2]+=1
        elif l<24:
            predicts[3]+=1
            #print(f"predicted LFG: {case}")
        elif l < 35:
            predicts[4]+=1
            #print(f"predicted MLFG: {case}")
        elif l == 35:
            predicts[5]+=1
    maxv= 0
    for i in range(len(predicts)):
        if predicts[i]>maxv:
            maxv=i
    if maxv == 0:
        out = "CMRG" 
    elif maxv==1:
         out = "LCG"
    elif maxv ==2:
         out = "LCG64"
    elif maxv==3:
         out = "LFG"
         #print(f"predicted LFG: {case}")
    elif maxv ==4:
         out = "MLFG"
         #print(f"predicted MLFG: {case}")
    elif maxv == 5:
         out = "PMLCG"
    return out
def main():
    df = pd.read_csv('output.csv')
    df = df.drop(["1","109"],axis = "columns")
    maxAccuracy = 0
    accls = []
    at = 0
    #for i in range(1,5):
    size = 0.2
    train_df, test_df = train_test_split(df, test_size=size, random_state=42,stratify=df['0'])
    column_name = '0'

#for i in range(90,100):
    confidence_level = 0.98
    labels = ["CMRG","LCG","LCG64","LFG","MLFG","PMLCG"]
    parameters = [3,7,3,11,11,1]
    numOfNANS = 0
    confidence_intervals_list = []
    for lab in range(0,len(labels)):
        
        subset_df = train_df[train_df[column_name] == labels[lab]]
        
        subset_df = subset_df.drop(columns=[column_name], axis='columns')
        for param in range(0,parameters[lab]):    
            furthersub = subset_df[subset_df["2"]==param]
            furthersub = furthersub.drop("2", axis ="columns")
            lil_ls = []
            localtest = 0
            for col in furthersub.columns:
                data = furthersub[col].dropna()
                #print(f"{col} {param} {lab}")
                confidence_interval = calculate_confidence_intervals(data, confidence_level)
                if np.isnan(confidence_interval).any():
                    localtest +=1
                    confidence_interval = (data.iloc[1]-(1-confidence_level),data.iloc[1]+(1-confidence_level))
                lil_ls.append(confidence_interval)
            if(localtest>0):
                numOfNANS +=1
            confidence_intervals_list.append(lil_ls)
    maxls=[]
    test_dfx = test_df.drop([column_name,"2"],axis="columns")
    actual = test_df[column_name].values
    for index, row in test_dfx.iterrows():
        templs = []
        maximumcorrect = 0
        predicted = 0
        #distanceoutsideofintervals = 0
        predicteddistance = 0
        #predictions = []
        for case in range(0,len(confidence_intervals_list)):
            modrow = row
            #print(modrow)
            distanceoutsideofintervals = 0
            correct = 0 
            for value in range(0,len(modrow)):
                #print(modrow[i])
                if confidence_intervals_list[case][value][0] <= modrow[value] and modrow[value]<=confidence_intervals_list[case][value][1]:
                    correct +=1
                elif confidence_intervals_list[case][value][0] > modrow[value]:
                    distanceoutsideofintervals+= confidence_intervals_list[case][value][0] - modrow[value]
                elif confidence_intervals_list[case][value][1] < modrow[value]:
                    distanceoutsideofintervals+= modrow[value]-confidence_intervals_list[case][value][1]
            #print(correct)
            templs.append(correct)
            if correct>maximumcorrect:
                maximumcorrect = correct
                predicted = casetoprediction(case)
                predicteddistance = distanceoutsideofintervals
                lsofsamecorrects = [case]
                #print(f"predictedDistance {predicteddistance}")
            elif correct == maximumcorrect:
                #print(distanceoutsideofintervals<predicteddistance)
                if predicteddistance > distanceoutsideofintervals:
                    predicted = casetoprediction(case)
                    predicteddistance = distanceoutsideofintervals
                    #print("gothere")
                lsofsamecorrects.append(case)
        #print(maximumcorrect)
        #print(templs)
        #pred1 = predictions(lsofsamecorrects)
        #maxls.append(pred1)
        maxls.append(predicted)
    #print(maxls)
    #print(actual)
    #print(numOfNANS)
    accuracy=(np.count_nonzero(maxls==actual)/len(maxls))#.count())#.count()/len(predictions))
    #print(i)
    conf = confusion_matrix(actual, maxls)
    print(len(maxls))
    accls.append(accuracy)
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        #at = i
    print(maxAccuracy)
    print(accls)
    print(at)
    print(conf)
    #print(len(confidence_intervals_list[13]))
        #print(confidence_intervals_list)
if __name__ == "__main__":
    main()