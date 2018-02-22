import math

# Utilities to calculate similariies


# Given two lists ordered by ascending indices, find matching indices
def findMatchingIndices(firstList, secondList):
    matching = []
    secondI = 0
    secondLen = len(secondList)
    for item in firstList:
        while secondI < secondLen and secondList[secondI] < item:
            secondI += 1
        if secondI >= secondLen:
            break
        if secondList[secondI] == item:
            matching.append(item)
    return matching


# Given two lists ordered by ascending indices, return a list of matching indices and lists of unique indices.
# Return matching, uniqueFirst, uniqueSecond
def findMatchingAndUniqueIndices(firstList, secondList):
    matching = []
    uniqueFirst = []
    uniqueSecond = []
    secondI = 0
    secondLen = len(secondList)
    firstI = 0
    for item in firstList:
        while secondI < secondLen and secondList[secondI] < item:
            # print("first: "+str(item)+" second: "+str(secondList[secondI]))
            uniqueSecond.append(secondList[secondI])
            secondI += 1
        if secondI >= secondLen:
            uniqueFirst.append(item)
            firstI += 1
            break
        if secondList[secondI] == item:
            matching.append(item)
            secondI += 1
        else:
            uniqueFirst.append(item)
        firstI += 1
    for itemI in range(secondI, secondLen):
        uniqueSecond.append(secondList[itemI])
    for itemI in range(firstI, len(firstList)):
        uniqueFirst.append(firstList[itemI])
    return matching, uniqueFirst, uniqueSecond


# Calculate cosine similarity given two doc vectors (parsing.Doc) and the precalculated sum of squares of the second vector.
def cosineSim(first, second):
    multSum = 0

    matching = findMatchingIndices(first.getUsedFeatures(), second.getUsedFeatures())
    for item in matching:
        multSum += first.getFeature(item) * second.getFeature(item)
        # print("multSum = "+str(multSum)+" after "+str(first.getFeature(item))+" * "+str(second.getFeature(item)))
    sim = multSum / (first.getSumSquares() * second.getSumSquares())
    # print("denominator: "+str((math.sqrt(first.getSumSquares() * second.getSumSquares()))))

    return sim


# Calculate cosine similarity given two doc vectors (parsing.Doc)
def euclideanDist(first, second):
    matching, uniqueFirst, uniqueSecond = findMatchingAndUniqueIndices(first.getUsedFeatures(), second.getUsedFeatures())
    distSum = 0
    for item in matching:
        distSum += (first.getFeature(item) - second.getFeature(item))**2
    for item in uniqueFirst:
        distSum += (first.getFeature(item))**2
    for item in uniqueSecond:
        distSum += (second.getFeature(item))**2
    return distSum # Don't take the square root, as we're just comparing scores.
