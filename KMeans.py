import collections

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# the distance function for a tables
def Distance(m,s):
    d = 0
    for i in range(len(m)):
        d = d + (m[i] - s[i]) **2
    return d**0.5

# function to find the new center by the average of their points
def RearrangeCenters(sets,start):

    for i in range(len(sets)):
        sx = 0
        sy = 0
        for j in range(len(sets[i])):
            sx = sx + sets[i][j][0]
            sy = sy + sets[i][j][1]
        if(len(sets[i]) != 0):
            start[i][0] = sx/len(sets[i])
            start[i][1] = sy / len(sets[i])
    return start

# function to find when the old and new centers have converged
def Converged(start,old):
    for i in range(len(start)):
        start[i][0] = round(start[i][0],3)
        start[i][1] = round(start[i][1], 3)
        old[i][0] = round(old[i][0], 3)
        old[i][1] = round(old[i][1], 3)
    for i in range(len(start)):
        if (start[i] != old[i]):
            return False
    return True

# MinMax function to initialize the starting points
def MinMaxStartingPoints(k,data):
    # convert data to data list
    data = data.tolist()
    # if k < 1 or data is empty
    if (k<1 or len(data)==0):
        return -1
    # the starting points
    start = []
    # the first starting point is the first element of our dataset
    start.append(data[0])

    # the rest k-1 starting points
    for i in range(1,k):
        # the second starting point is the furthest from the first one
        startPoint = max(data , key=lambda data:min(Distance(data,x) for x in start))
        data.remove(startPoint)
        start.append(startPoint)
    # return the k starting points
    return start

# k the number of neighborhoods
# M the data that we want to cluster
# the number of the max iterations
def KMeans(k,M,maxIterations):
    # the first element of the M^ table to separate list
    x = []
    for i in range(len(M)):
        x.append(M[i][0])

    # the second element of the M^ table to separate list
    y = []
    for i in range(len(M)):
        y.append(M[i][1])

    # call the MinMax function to initialize the starting points
    start = MinMaxStartingPoints(k,M)

    # a boolean variable to find if the clusters have converged
    convergedBool = False

    # the number of itarations needed
    iterations = 0

    while( convergedBool == False and iterations < maxIterations):
        iterations = iterations +1

        # clusters sets, first for 0, second for 1 etc.
        sets = [[] for _ in range(k)]

        # assign points to their closest centers
        for i in range(len(M)):

            minDis = Distance(M[i],start[0])
            cluster = 0
            for j in range(len(start)):
                if (Distance(M[i],start[j]) < minDis):
                    minDis = Distance(M[i],start[j])
                    cluster = j
            sets[cluster].append(M[i].tolist())

        # save the old starting points
        old = copy.deepcopy(start)
        # compute the new starting points
        start = RearrangeCenters(sets,start)
        # chech to find if they have converged
        convergedBool = Converged(start,old)

    #print("Iterations taken ",iterations)
    # return the sets
    return sets

# s : the specified set we want to find the number of correct points
# M : the lists with all the points , starting with the one of class 0 etc.
# lenghts : the lenght of each class of M
# the number of classes
def findMaxCorrectPoints(set, M,Ltr, lengths,n):

    # for the clusters we count how many points are assigned correctly
    commonPoints = [0] * n

    # for each element in the Set we find to which of the N classes belongs to
    # and we keep count

    for element in set:
        number = -1
        if element in M[:lengths[0]]:
            number = 0
        elif element in M[lengths[0] : lengths[1]]:
            number = 1
        elif element in M[lengths[0] + lengths[1] : lengths[2]]:
            number = 2
        else:
            number = 3
        commonPoints[number] +=1


    # we have N values with the number of assigned points for this set
    # since we consider the one with the most points to be the correct
    # one we return the max value of these N numbers
    return max(commonPoints)


# function to find the purity of the sets
def getPurity(M,Ltr,sets,numberOfClasses):
    # the length of each class in M list
    lengths = [0] * numberOfClasses
    for x in Ltr:
        if x == 0:
            lengths[0] = lengths[0] + 1
        elif x == 1:
            lengths[1] = lengths[1] + 1
        elif x == 2:
            lengths[2] = lengths[2] + 1
        elif x == 3:
            lengths[3] = lengths[3] + 1

    sum = 0
    for set in sets:
        points = findMaxCorrectPoints(set,M,Ltr,lengths,numberOfClasses)
        sum += points

    return sum/len(M)



def plot(sets,k):
    # save the M list to two lists x and y with the first and second element
    x = [[] for _ in range(k)]
    y = [[] for _ in range(k)]

    for j in range(k):
        for i in range(len(sets[j])):
            x[j].append(sets[j][i][0])
            y[j].append(sets[j][i][1])

    # plot the sets, use the 4 standard colors for the first 4 sets
    # and a random color for the nexts ones
    # I used alpha = 0.3 to see the areas with more concatrated points
    plt.style.use('ggplot')

    for i in range(k):
        if i == 0:
            color = 'r'
        elif i == 1:
            color = 'g'
        elif i == 2:
            color = 'c'
        elif i == 3:
            color = 'y'
        else:
            color = '#%06X' % random.randint(0, 0xFFFFFF)
        plt.scatter(x[i], y[i], color=color, alpha=0.3)

    # add grid
    plt.grid(True)
    plt.title("All the numbers after the K-Means classification for K = 4")

    plt.show()

def main():
    # read the M^ and Ltr file from exercise 2

    M = pd.read_csv("Files/MD.csv").values

    Ltr = pd.read_csv("Files/Ltr.csv").values

    # number k
    k = 4
    # calling the K Means function
    sets = KMeans(k, M, 1000)

    # the nubmer of the classes
    numberOfClasses = 4


    # get the purity
    purity = getPurity(M, Ltr, sets, numberOfClasses)
    print("Purity is ", purity)

    plot(sets,k)
















