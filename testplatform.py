# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:13:45 2019

@author: Xiao Chi
"""

import re
import epilearn
import datetime
import time

###############################################################################
###############################################################################
#from utils import validation_for_genandtest


def runbatch(tupleset,repetitions,dataname,constraintset,relation):
    for mytuple in tupleset:
        repeatrun(mytuple,repetitions,dataname,constraintset,relation)
        
        
        
        
def repeatrun(mytuple,repetitions,dataname,constraintset,relation):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    posscols = mytuple[0]
    posshead = mytuple[1]
    sumtime = 0
    sumrules = 0
    sumcover = 0
    sumaccuracy = 0
    sumlift = 0
    sumconditions = 0
    for x in range(0,repetitions):
        print(str(datetime.datetime.now().time()))
        t0 = time.time()
        scores = runtuple(mytuple,dataname,constraintset,relation)
        t1 = time.time()
        sumtime = sumtime+(t1-t0)
        sumrules = sumrules + scores[0]
        sumcover = sumcover + scores[1]
        sumaccuracy = sumaccuracy + scores[2]
        sumlift = sumlift + scores[3]
        sumconditions = sumconditions + scores[4]
        simplebestrules = scores[5]
    avtime = doaverage(sumtime,repetitions)
    avrules = doaverage(sumrules,repetitions)
    avcover = doaverage(sumcover,repetitions)
    avaccuracy = doaverage(sumaccuracy,repetitions)
    avlift = doaverage(sumlift,repetitions)
    avconditions = doaverage(sumconditions,repetitions)
    print("\n")
    print("Head = "+str(posshead))
    print("Tail = "+str(posscols))
    print("\n")
    print("Number of repetitions \t = "+str(repetitions))    
    print("Av time per rule set \t = "+str(round(avtime,2)))    
    print("Av number of rules \t = "+str(round(avrules,2)))
    print("Av cover of rules \t = "+str(round(avcover,2)))
    print("Av accuracy of rules \t = "+str(round(avaccuracy,2)))
    print("Av lift of rules \t = "+str(round(avlift,2)))
    print("Av conditions per rule \t = "+str(round(avconditions,2)))    
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print("\n")
    for x in simplebestrules:
        print(str(x))
        print("\n")
    print("\n")
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        

def runtuple(mytuple,dataname,constraintset,relation):
    if dataname == "spain":
        filename = "wiki4HE.csv"
    elif dataname == "italy":
        filename = "politicalattitudes.csv"
    posscols = mytuple[0]
    posshead = mytuple[1]
    results = epilearn.genandtest(filename,posscols+posshead,posshead,dataname,constraintset,relation)
    return results
    

def doaverage(x,y):
    if x > 0 and y > 0:
        return x/y
    else:
        return 0
    
    
###############################################################################
###############################################################################

def run(code):
    if code == 1: 
        dataname = "italy"
        source = ["sys1","sys2"]
        target = ["sys3"]
        relation = [0,0]
        constraintset = [0.5]
    
    if code == 2:
        dataname = "spain"
        source = ["SA2","SA1"] 
        target = ["SA3"]
        relation = [1,1]
        constraintset = [0.5]
    
    if code == 3: 
        dataname = "italy"
        source = ["dw4","dw3","dw7"]
        target = ["dw1"]
        relation = [1,0,1]
        constraintset = [0.5]
    
    if code == 4:
        dataname = "spain"
        source = ["SA2","Use3","Use1"] 
        target = ["Use2"]
        relation = [1,1,1]
        constraintset = [0.5]
    
    if code == 5: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8"]
        target = ["dw2"]
        relation = [0,1,0,1]
        constraintset = [0.5]
        
    if code == 6:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4"] 
        target = ["Use2"]
        relation = [1,1,1,1]
        constraintset = [0.5]

    if code == 7: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1"]
        target = ["dw2"]
        relation = [0,1,0,1,0]
        constraintset = [0.5]
    
    if code == 8:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1"] 
        target = ["Use2"]
        relation = [1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 9: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0]
        constraintset = [0.5]

    if code == 10:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1]
        constraintset = [0.5]

    if code == 11: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1]
        constraintset = [0.5]
    
    if code == 12:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 13: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0]
        constraintset = [0.5]
    
    if code == 14:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 15: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1]
        constraintset = [0.5]
    
    if code == 16:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 17: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0]
        constraintset = [0.5]

    if code == 18:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]

    if code == 19: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0]
        constraintset = [0.5]   
    
    if code == 20:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 21: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0,1]
        constraintset = [0.5]   
    
    if code == 22:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 23: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0,1,0]
        constraintset = [0.5]   
    
    if code == 24:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 25: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0]
        constraintset = [0.5]  
    
    if code == 26:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1","Qu2"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 27: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0,0]
        constraintset = [0.5]  

    if code == 28:
        dataname = "spain"
        source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1","Qu2","Qu3"] 
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 29: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6","sys7"]
        target = ["dw2"]
        relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1]
        constraintset = [0.5]
    
    if code == 30:
        dataname = "spain"
        source = ["BI1","BI2","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1"]
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    if code == 31: 
        dataname = "italy"
        source = ["dw4","dw3","dw7","dw8","dw2","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6","sys7","sys8"]
        target = ["dw1"]
        relation = [1,0,1,0,0,1,0,1,0,1,1,0,1,1,1,0,1]
        constraintset = [0.5]
    
    if code == 32:
        dataname = "spain"
        source = ["BI1","BI2","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]
        target = ["Use2"]
        relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        constraintset = [0.5]
    
    # if code == 1: 
    #     dataname = "italy"
    #     source = ["sys1","sys2"]
    #     target = ["sys3"]
    #     relation = [0,0]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 2:
    #     dataname = "spain"
    #     source = ["SA2","SA1"] 
    #     target = ["SA3"]
    #     relation = [1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 3: #2
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7"]
    #     target = ["dw1"]
    #     relation = [1,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 4:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 5: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
        
    # if code == 6:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 7: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 8:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 9: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 10:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 11: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 12:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 13: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 14:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 15: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 16:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 17: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 18:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 19: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0]
    #     constraintset = [0.25, 0.5, 0.75]   
    
    # if code == 20:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 21: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0,1]
    #     constraintset = [0.25, 0.5, 0.75]   
    
    # if code == 22:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 23: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0,1,0]
    #     constraintset = [0.25, 0.5, 0.75]   
    
    # if code == 24:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 25: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0]
    #     constraintset = [0.25, 0.5, 0.75]  
    
    # if code == 26:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1","Qu2"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 27: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0,0]
    #     constraintset = [0.25, 0.5, 0.75] 

    # if code == 28:
    #     dataname = "spain"
    #     source = ["SA2","Use3","Use1","Use4","SA1","SA3","JR1","JR2","Im1","Im2","ENJ1","ENJ2","Qu1","Qu2","Qu3"] 
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 29: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw1","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6","sys7"]
    #     target = ["dw2"]
    #     relation = [0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 30:
    #     dataname = "spain"
    #     source = ["BI1","BI2","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1"]
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 31: 
    #     dataname = "italy"
    #     source = ["dw4","dw3","dw7","dw8","dw2","dw5","dw6","dw9","dw10","sys1","sys2","sys3","sys4","sys5","sys6","sys7","sys8"]
    #     target = ["dw1"]
    #     relation = [1,0,1,0,0,1,0,1,0,1,1,0,1,1,1,0,1]
    #     constraintset = [0.25, 0.5, 0.75]
    
    # if code == 32:
    #     dataname = "spain"
    #     source = ["BI1","BI2","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]
    #     target = ["Use2"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]

    # if code == 33:
    #     dataname = "spain"
    #     source = ["BI1","BI2","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]
    #     target = ["Use1"]
    #     relation = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #     constraintset = [0.25, 0.5, 0.75]

        
    tupleset = [[source,target]]
    repetitions = 1
    runbatch(tupleset,repetitions,dataname,constraintset,relation)
     
###############################################################################
###############################################################################


if __name__ == "__main__":
    # codes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    # for code in codes:
    #     run(codes)
    run(6)
    


###############################################################################
###############################################################################