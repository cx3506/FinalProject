####################################################
####################################################
####################################################
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:40:22 2019

Tues 17 Sept 

@author: Tony
"""

####################################################
import spaintablemanager
import italytablemanager
import random



####################################################
####################################################
####################################################

'''
Generate a proper dataset with probability values and split it into train and test sets


'''
def genandtest(filename,posscols,possheads,dataname):
    if dataname == "spain":
        mydictionary = spaintablemanager.dodatadictionary(filename) # all data
    elif dataname == "italy":
        mydictionary = italytablemanager.dodatadictionary(filename)        
    selecteddata = selectdata(mydictionary,posscols) # data of selected cols 
    improveddata = removequestiondata(selecteddata,posscols) # Deleting rows with "?"   
    dataset = transform(improveddata,posscols)  # Probability
    splitdata = gettraintestsets(dataset) # split data into train and test sets
    trainingdata = splitdata[0] 
    testingdata = splitdata[1]
    allbestrules = []
    rulecount = 0
    for head in possheads:
        cols = posscols[:]
        if head in cols:
            cols.remove(head)
        allsets = genpowerset(cols) # posscols的排列组合
        for myset in allsets:
            if myset != []  and len(myset) < 4:
                reqcols = [head]+myset
                rules = genformulaset(reqcols,trainingdata) # generate a set of rules 
                rulecount = rulecount+len(rules) #num of rules
                newrules = improveformulaset(reqcols,trainingdata,rules) #Generate a new rule set with rules comparing to 0.5 
                bestrules = choosebest(reqcols,trainingdata,newrules) # Obtain best rules from newrules
                allbestrules = allbestrules+bestrules # Set of all best rules after the whole loop
    simplebestrules = choosesimplest(allbestrules) #Generate a set of simplest rules from all bestrules                
    sumcover = 0
    sumaccuracy = 0
    sumlift = 0
    sumconditions = 0
    numberofrules = len(simplebestrules) #number of simplest best rules
    for rule in simplebestrules:
        scores = getrulequality(reqcols,testingdata,rule) # ratio of correct and fired (fired,correct)
        testcoverscore = scores[0]
        testaccuracyscore = scores[1]
        testliftscore = getliftscore(reqcols,testingdata,rule) #Calculate lift score
        numberofconditions = getnumberofconditions(rule) #
        sumcover = sumcover + testcoverscore
        sumaccuracy = sumaccuracy + testaccuracyscore
        sumlift = sumlift + testliftscore
        sumconditions = sumconditions + numberofconditions
    if sumcover > 0:
        avcover = sumcover/numberofrules
    else: 
        avcover = 0
    if sumaccuracy > 0:
        avaccuracy = sumaccuracy/numberofrules
    else:
        avaccuracy = 0
    if sumlift > 0:    
        avlift = sumlift/numberofrules
    else:
        avlift = 0
    if sumconditions > 0:        
        avconditions = sumconditions/numberofrules
    else:
        avconditions = 0
    allscores = [numberofrules,avcover,avaccuracy,avlift,avconditions,simplebestrules] 
    print(str(rulecount))
    return allscores








####################################################
####################################################
####################################################


'''
排列组合
'''
def genpowerset(X):
    allsets = [[]]
    for x in X:
        newsets = []
        for s in allsets:
            t = s[:]
            t.append(x)
            newsets.append(t)
        allsets = allsets+newsets
    return allsets
    
'''
Deleting rows with '?' 
'''   
def removequestiondata(selecteddata,reqcols):
    improveddata = []
    for mydictionary in selecteddata:
        flag = "keep"
        for col in reqcols:
            if mydictionary[col] == "?":
                flag = "delete"
        if flag == "keep":
            improveddata.append(mydictionary)
    return improveddata
    
 
       
####################################################
####################################################
####################################################
# SELECT DATA

'''
select data from database
'''        
def selectdata(database,reqcols):
    selected = []
    for dictionary in database:
        sub = {}
        sub["myid"] = dictionary["myid"] 
        for col in reqcols:
            sub[col] = dictionary[col]
        #print(str(sub))
        selected.append(sub)
    return selected


'''
split selecteddata into traindata and testdata
'''
def gettraintestsets(selecteddata):
    alldatacount = len(selecteddata)
    requirement = 0.2 * alldatacount
    traindata = []
    testdata = []
    for item in selecteddata: 
        act = random.randrange(10)
        if act < 2 and len(testdata) < requirement:
            testdata.append(item)
        else:
            traindata.append(item)
    return [traindata,testdata]


####################################################
####################################################
####################################################
# TRANSFORM INTO DATASET
# Tuples with prob values

'''
Trasform  actual values into probability values
'''
def transform(selecteddata,reqcols):
    transformdata = []
    for x in selecteddata:
        y = {}
        y["myid"] = x["myid"]
        for col in reqcols:
            short = col[:2]
            #print("short = "+short)
            if  short == "PO" or short == "cb":
                #print("change7")
                newvalue = change7(x[col])
            elif short == "sy" or short == "sd":
                #print("change8")
                newvalue = change8(x[col])
            elif short == "cj" or short == "rw" or short == "dw":
                #print("change10")
                newvalue = change10(x[col])
            else:
                #print("col = "+str(col)+"\tvalue = "+str(x[col]))
                newvalue = change(x[col])
            y[col] = newvalue
        transformdata.append(y)
        #print(str(y))
    return transformdata


def change(v):
    if v == "1":
        p = 0.1
    elif v == "2":
        p = 0.3
    elif v == "3":
        p = 0.5
    elif v == "4":
        p = 0.7
    elif v == "5":
        p = 0.9
    elif v == "?":
        p = "?"
    return p

def change7(v):
    if v == "1":
        p = 0
    elif v == "2":
        p = 0.1
    elif v == "3":
        p = 0.3
    elif v == "4":
        p = 0.5
    elif v == "5":
        p = 0.7
    elif v == "6":
        p = 0.9
    elif v == "7":
        p = 1.0
    return p      

def change8(v):
    if v == "1":
        p = 0
    elif v == "2":
        p = 0.2
    elif v == "3":
        p = 0.3
    elif v == "4":
        p = 0.4
    elif v == "5":
        p = 0.6
    elif v == "6":
        p = 0.7
    elif v == "7":
        p = 0.8
    elif v == "8":
        p = 1.0
    return p        

        
def change10(v):
    if v == "1":
        p = 0.1
    elif v == "2":
        p = 0.2
    elif v == "3":
        p = 0.3
    elif v == "4":
        p = 0.4
    elif v == "5":
        p = 0.5
    elif v == "6":
        p = 0.6
    elif v == "7":
        p = 0.7
    elif v == "8":
        p = 0.8
    elif v == "9":
        p = 0.9
    elif v == "10":
        p = 1.0
    return p        

####################################################
####################################################
####################################################
# FORMULA GENERATOR

'''
Generate a formula
'''
def genformula(argnames,data):
    print(argnames);
    print(data);
    head = [argnames[0],"E",data[argnames[0]]]
    #print(str(head))
    tail = ""
    for n in range(1,len(argnames)):
        conjunct = [argnames[n],"E",data[argnames[n]]]
        #print(str(conjunct))
        if n == 1:
            tail = conjunct
        else:
            tail = [conjunct,"AND",tail]
    #print(str(tail))
    formula = [tail,"IMPLIES",head]
    #print(str(formula))
    return formula
    
'''
generate a set of formulas from a dataset
'''  
def genformulaset(argnames,dataset):
    formulaset = []
    print(dataset); 
    print("1");
    for data in dataset:
        formula = genformula(argnames,data)
        #print("\n")
        #print(formula)
        formulaset.append(formula)
    return formulaset


####################################################
####################################################
####################################################
# FORMULAE IMPROVEMENT
# Given a set of formulae, 
# this method  determines the quality of each rule,
# and then determines which rules should be removed
# and which rules should be generalized or specialized

# Note since there are mutliple options, 
# we have variantes of this method

'''
Generalize a new formulaset with tail and head compared to 0.5, and 
all rules in the set are not the same
'''
def improveformulaset(reqcols,dataset,rules):
    generalrules = headgeneralize(rules)  #generalize new rules with new head
    moregeneralrules = tailgeneralize(generalrules) # 
    return deduplicate(moregeneralrules)
    
'''
return formulae in formulae that are not the same
'''
def deduplicate(formulae):
    new = []
    for formula in formulae:
        if notequalmember(formula,new):
            new.append(formula)
    return new

'''
return not equal formula
'''
def notequalmember(formula,new):
    for other in new:
        if equalcheck(formula,other):
            return False
    return True


'''
Check whether rule1 and rule2 are the same
'''    
def equalcheck(form1,form2):
    if str(form1) == str(form2):
        return True
    else:
        return False
        
        
        

####################################################
####################################################
####################################################
# GENERALIZATION
        
'''
Generalize newrules with new tail
'''    
def tailgeneralize(rules):
    newrules = []
    for formula in rules:
        newrule = tailgen(formula)
        newrules.append(newrule)
    return newrules

'''
Change the rule by comparing the tail with 0.5
'''      
def tailgen(formula):
    i = 0
    #print("Formula[0] = "+ str(formula[0]))
    tail = getconjuncts(formula[0])
    #print("Tail = "+str(tail))
    while i < len(tail):
        condition = tail[i]
        newcondition = condition[:]
        #print("Condition = "+str(condition))
        if condition[2] > 0.5:
            newcondition[0] = condition[0]
            newcondition[1] = "G"
            newcondition[2] = 0.5
        elif condition[2] <= 0.5:            
            newcondition[0] = condition[0]
            newcondition[1] = "LE"
            newcondition[2] = 0.5
        tail[i] = newcondition
        i = i+1
    #print("Tail = "+str(tail))
    conjoined = formconjunction(tail)
    #print("Conjoined = "+str(conjoined))
    formula[0] = conjoined
    return formula
    
'''
Generalize newrules with new head
'''
def headgeneralize(rules):
    newrules = []
    for formula in rules:
        newrule = headgen(formula)
        newrules.append(newrule)
    return newrules

'''
Change the rule by comparing the head with 0.5
e.g. BI1<0.3 to BI1<0.5, BI1 is the head of the rule
'''    
def headgen(formula):
    head = formula[2]
    newformula = []
    newhead = []
    if (head[1] == "E") and float(head[2]) > 0.5:
        newhead.append(head[0])
        newhead.append("G")
        newhead.append(0.5) 
    elif (head[1] == "L" or head[1] == "E") and float(head[2]) <= 0.5:
        newhead.append(head[0])
        newhead.append("LE")
        newhead.append(0.5)
    else:
        newhead.append(head[0])
        newhead.append(head[1])
        newhead.append(head[2])        
    #print("New head = "+str(newhead))
    newformula.append(formula[0])
    newformula.append(formula[1])
    newformula.append(newhead)
    #print("New formula = "+str(formula))
    return newformula
    
    
    
        
        
   
####################################################
####################################################
####################################################
# return rules with min condition
# so rules with more specialized condition are rejected

'''
Generate a set of simplest rules
'''
def choosesimplest(rules):
    simplest = []
    for rule in rules:
        #print(str(rule))
        if mincondition(rule,rules):
            #print("Yes")
            simplest.append(rule)
    return simplest

'''
Return the simpler rule with the same influence target
'''
def mincondition(rule,rules):
    for other in rules:
        if rule[2] == other[2] and set(atoms(other[0])) < set(atoms(rule[0])):        
            return False
    return True

'''
return influencers
i.e. 'PU3'
'''    
def atoms(con):
    if con[1] != "AND":
        return [con[0]]
    else:
        return atoms(con[0])+atoms(con[2])
        
     
    
####################################################
####################################################
####################################################
# Choose best rules based on 
# coverage, accuracy, etc

'''
Generate best rules out of a set of rules
'''    
def choosebest(reqcols,dataset,rules):
    best = []
    for rule in rules:
        #print("\n"+str(rule))
        scores = firescore(reqcols,dataset,rule)
        #print(str(scores))
        correct = scores[0] # number of rules that are correct
        incorrect = scores[1]
        nocoverage = scores[2] 
        coverage = len(dataset) - nocoverage #number of rules that are fired
        goodcover = coverageratio(dataset,rule,coverage) #Support(R,D)
        goodcorrect = correctratio(correct,incorrect) #Confidence(R,D)
        liftscore = getliftscore(reqcols,dataset,rule) #Lift(R,D)
        #print("lift = "+str(liftscore))
        #if goodcorrect and liftscore > 2:        
        if goodcover and goodcorrect and liftscore > 1:
            #print("best")
            best.append(rule)
    return best

'''
Calculate Support(R,D)
'''
def coverageratio(dataset,rule,coverage):
    coverratio = float(coverage)/len(dataset)
    #print("coverratio = "+str(coverratio))
    #if coverratio > 0.1:    
    if coverratio > 0.4:
    #if coverratio > 0.05:
        return True
    else:
        return False
    
'''
Calculate Confidence(R,D)
'''        
def correctratio(correct,incorrect):
    #print("x")
    correctratio = float(correct)/(incorrect+correct)
    #print("correctratio = "+str(correctratio))
    #if correctratio > 0.6:
    if correctratio > 0.8:
        return True
    else:
        return False
     

'''
Calculate the ratio for fired and correct
'''
def getrulequality(reqcols,dataset,rule):
    scores = firescore(reqcols,dataset,rule)
    correct = scores[0]
    incorrect = scores[1]
    nocoverage = scores[2]
    coverage = len(dataset) - nocoverage #number of rules that are fired
    coverratio = float(coverage)/len(dataset) #ratio of coverage
    #print("coverratio = "+str(coverratio))
    correctratio = float(correct)/(incorrect+correct)  #ratio of correct 
    #print("correctratio = "+str(correctratio))
    return [coverratio,correctratio]

####################################################
####################################################
####################################################
# SCORING A DATASET
    
'''
correct(Y): number of rules that are correct
incorrect(N): number of rules that are fired but not agrees:
nofire(NA): number of rules that are not fired
'''
def firescore(argnames,dataset,formula):    
    correct = 0
    incorrect = 0
    nofire = 0
    #print("\n\n")
    for data in dataset:
        reply = satformula(argnames,data,formula)
        #print(reply +"\t"+str(str(data)))
        if  reply == "Y":
            correct = correct + 1
        elif  reply == "N":
            incorrect = incorrect + 1
        elif  reply == "NA":
            nofire = nofire + 1
    return [correct,incorrect,nofire]


def satcount(argnames,dataset,formula):
    count = 0
    for data in dataset:
        reply = satformula(argnames,data,formula)
        if  reply == "Y" or reply == "NA":
            count = count + 1
    return count

'''
Calculate number of correct rules
'''
def correctfirecount(argnames,dataset,formula):
    count = 0
    for data in dataset:
        reply = satformula(argnames,data,formula)
        if  reply == "Y":
            count = count + 1
    return count


def sumerror(argnames,dataset,formula):
    sumerror = 0
    for data in dataset:
        error = errorformula(argnames,data,formula)
        sumerror = error + sumerror
    return sumerror
   
def firecount(argnames,dataset,formula):
    count = 0
    for data in dataset:
        if sattail(argnames,data,formula[0]):
            count = count + 1
    return count



####################################################
####################################################
####################################################

'''
Calculate lift
'''
def getliftscore(argnames,dataset,rule):
    f = numberfirings(argnames,dataset,rule) #number of fired rules
    #print("\t f ="+str(f))
    e = numbererrorfree(argnames,dataset,rule) #number of agreed rules
    #print("\t e ="+str(e))
    g = correctfirecount(argnames,dataset,rule) #number of correct rules 
    #print("\t g ="+str(g))
    n = len(dataset)
    #print("\t n ="+str(n))
    lift = ((g * n ) / (e * f))
    return lift
    
'''
Calculate the number of fied rules
'''    
def numberfirings(argnames,dataset,rule):
    ftuple = firescore(argnames,dataset,rule)
    return ftuple[0] + ftuple[1]

'''
Total number of agreed rules
'''
def numbererrorfree(argnames,dataset,formula):
    count = 0
    for data in dataset:
        error = errorhead(argnames,data,formula)
        if error == 0:
            count = count + 1
    return count

'''
Decide whether a rule agrees
if true, return 0
'''
def errorhead(argnames,data,formula):
    head = formula[2]
    if data[head[0]] == "?" or head[2] == "?":
        error = 0
    else:
        if head[1] == "E":
            error = diff(float(data[head[0]]),float(head[2]))
        elif head[1] == "G" or head[1] == "GE":
            error = below(float(data[head[0]]),head[2])
        elif head[1] == "L" or head[1] == "LE":
            error = above(float(data[head[0]]),head[2])
    return error    

####################################################
####################################################
####################################################
# SATISFACTION WRT A DATATUPLE
        
'''
Correct: Y
fired not agrees: N
neither fired nor agrees: NA
'''        
def satformula(argnames,data,formula):
    if sattail(argnames,data,formula[0]):
        #print("satformula = "+str(formula[0]))
        if sathead(argnames,data,formula[2]):
            return "Y"
        else:
            return "N"
    else:
        return "NA"
        
'''
Decide fired (tails)
If fired, return True
If not, return False
'''        
def sattail(argnames,data,tail):
    conjuncts = getconjuncts(tail)
    #print("conjuncts = "+str(conjuncts))
    for atom in conjuncts:
        #print("sattail = "+str(tail))
        if nonsatatom(argnames,data,atom):
            #print("nonsattail")
            return False
    return True

'''
Decide agrees (head)
If agrees, return True
If not, return False
'''
def sathead(argnames,data,head):
    if nonsatatom(argnames,data,head):
        return False
    else:
        return True

    

'''
If fired or agrees, return False
If not, return True
'''
def nonsatatom(argnames,data,atom):
    if atom[1] == "E":
        if float(data[atom[0]]) == atom[2]:
            return False
        else:
            return True
    if atom[1] == "G":
        if float(data[atom[0]]) > atom[2]:
            return False
        else:
            return True
    if atom[1] == "GE":
        if float(data[atom[0]]) >= atom[2]:
            return False
        else:
            return True
    if atom[1] == "L":
        if float(data[atom[0]]) < atom[2]:
            return False
        else:
            return True
    if atom[1] == "LE":
        if float(data[atom[0]]) <= atom[2]:
            return False
        else:
            return True
        
          
        
def errorformula(argnames,data,formula):
    error = 0
    #print(str(formula))
    if sattail(argnames,data,formula[0]):
        #print("satformula = "+str(formula[0]))
        head = formula[2]
        if data[head[0]] == "?" or head[2] == "?":
            error = 0
        else:
            if head[1] == "E":
                error = diff(float(data[head[0]]),float(head[2]))
            elif head[1] == "G" or head[1] == "GE":
                error = below(float(data[head[0]]),head[2])
            elif head[1] == "L" or head[1] == "LE":
                error = above(float(data[head[0]]),head[2])
    return error
            
'''
Calculate the absolute difference between x and y
'''             
def diff(x,y):
    if x == y:
        return 0
    elif x > y:
        return x - y
    else:
        return y - x
    
# Following method conflates G or GE error 
 
def below(x,y):
    #datavalue x needs to be above y
    #so error is amount that x is below 
    if y > x:
        return y - x
    else:
        return 0


def above(x,y):
    #print("x,y = "+str(x)+"/"+str(y))
    if x > y:
        return x - y
    else:
        return 0


####################################################
####################################################
####################################################
# Data points that covered correctly by a rule

def getcover(reqcols,dataset,rule):
    cover = []
    for data in dataset:
        reply = satformula(reqcols,data,rule)
        if  reply == "Y":
            cover.append(data)
    return cover

    



####################################################
####################################################
####################################################

'''
Add 'And' 
'''
def formconjunction(conjuncts):
    if len(conjuncts) == 1:
        return conjuncts[0]
    else:
        return [conjuncts[0],"AND",formconjunction(conjuncts[1:])]
    
    
    
'''
get rid of AND
e.g. 'a>0.5', 'and', 'b<0.3' to 'a>0.5', 'b<0.3'
'''  
def getconjuncts(tail):
    if tail[1] == "AND":
        return [tail[0]] + getconjuncts(tail[2]) 
    else:
        return [tail]


'''
Calculate the number of conditions for the rule
'''
def getnumberofconditions(rule): 
    rulestring = str(rule)
    andcount = rulestring.count("AND")
    return andcount+1
    
    
####################################################
####################################################
####################################################

if __name__ == "__main__":
    filename="wiki4HE.csv"
    dataname = "spain"
    posscols = ["PU3","Qu1","Qu3","ENJ1"]
    posshead = ['BI1']
    #allscores = genandtest(filename,posscols,possheads,dataname)
    allscores = genandtest(filename, posscols + posshead, posshead, dataname)
    
    
    
####################################################
####################################################
####################################################
    
#argnames = ["a1","a2"]

#dataset = [[1,1],[1,1],[1,1]]

#datatuple = [1,0]

#atom = ['a2', 'G', 0]

#f1 = [[['a2', 'E', 1],'AND',['a3','E',0.3]], 'IMPLIES', ['a1', 'E', 0.9]]

#f1 = [['a2', 'E', 1], 'IMPLIES', ['a1', 'E', 0.9]]


#f2 = [['a2', 'E', 1], 'IMPLIES', ['a1', 'E', 1]]

#tf = nonsatatom(argnames,datatuple,formula)

#tf = satformula(argnames,datatuple,formula)

#print(str(tf))


#c = correctfirecount(argnames,dataset,formula)

#c =  sumerror(argnames,dataset,formula)

#print(str(c))

#print(str(f1))
#print(str(headgen(f1)))

#f = [[['a4', 'E', 0.5], 'AND', [['a3', 'E', 0], 'AND', ['a2', 'E', 0]]], 'IMPLIES', ['a1', 'E', 1]]

#print(str(getconjuncts(f[0])))

#argnames = ["a1","a2","a3","a4"]

#datatuple = [1,0,1,0.5]

#dataset = [datatuple]

#formulae = genformulaset(argnames,dataset)

#for formula in formulae:
#    print(str(formula))
    
    
#f0 = [[['a3', 'E', 0], 'AND', ['a2', 'E', 0]], 'IMPLIES', ['a1', 'E', 1]]    
#f1 = [[['a4', 'E', 0.5], 'AND', [['a3', 'E', 0], 'AND', ['a2', 'E', 0]]], 'IMPLIES', ['a1', 'E', 1]]
#f2 = [[['a4', 'E', 0.5], 'AND', [['a3', 'E', 1], 'AND', ['a2', 'E', 0]]], 'IMPLIES', ['a1', 'E', 1]]
#equalcheck(f1,f2)

#print(str(biggercondition(f0,f1)))

#filename = "wiki4HE.csv"
#
#reqcols = ["PU3","Qu1","ENJ1"]
#reqcols = ["PU1","Qu1","Qu3","ENJ1"]
#reqcols = ["PU3","Qu1","Qu3","ENJ1"]
#reqcols = ["PU3","Qu1","Qu3","ENJ1","JR1","JR2","SA1"]
#reqcols = ["PU3","Qu1"]
#reqcols = ["PU1","Qu1"]
#reqcols = ["BI1","Use2","PU3","Im2"]
#argnames = ["a1","a2","a3","a4"]
#pi = [0,0.25,0.5,0.75,1]

#main(filename,reqcols)

#posscols = ["ENJ1","ENJ2"]
#posscols = ["PU1","PU2","PU3"]
#posscols = ["Pf1","Qu1","PEU1"]
#posscols = ["PU3","Qu1","Qu3","ENJ1"]
#posscols = ["Qu1","Qu3","ENJ1"]
#posscols = ["PU3","Qu1"]
#posscols = ["Qu1","Qu3","ENJ1","JR1","JR2","SA1"]
#posscols = ["BI1","PU1","PU2","PU3","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]

#posscols = ["JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]
#posscols = ["JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2"]

#posscols = ["BI1","PU1","PU2","PU3","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]#posscols = ["BI1","PU3","JR1","SA1","Im1","Pf1","Qu1","ENJ1","PEU1"]
#posscols = ["BI1","BI2","Use1","Use2","Use3","Use4"]
#posscols = ["BI1","BI2"]
#posscols = ["PU1","PU2","PU3","JR1","JR2","SA1","SA2","SA3","Im1","Im2","Pf1","Pf2","Pf3","Qu1","Qu2","Qu3","ENJ1","ENJ2","PEU1","PEU2"]
#head = "PU3"
#supermain(filename,posscols,head)
#possheads = ["BI2"]
#posscols = ["PU1","PU2","PU3","SA1","SA2","SA3","Im1","Im2"]
#possheads = ["PEU2"]
#possheads = ["PU1"]
#ossheads = ["Use1","Use2","Use3","Use4"]
#possheads = ["Use2"]

#newmain(filename,posscols+possheads,possheads)
        
#c = [['a4', 'E', 0.5], 'AND', [['a3', 'E', 0], 'AND', ['a2', 'E', 0]]]    
    
#print(str(atoms(c)))

#mydictionary = tablemanager.dodatadictionary(filename)
#
#for x in mydictionary:
#    print(str(x["PEU2"]))
#

####################################################
####################################################
####################################################
####################################################
####################################################