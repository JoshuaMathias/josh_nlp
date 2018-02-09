import similarities
import parsing
from scipy import spatial
import numpy as np
import math


# similarities.findMatchingIndices
first = [1, 3, 5, 6, 7]
second = [0, 4, 6, 7]
correctMatches = [6, 7]
foundMatches = similarities.findMatchingIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)

first = [1, 3, 5, 6, 7, 8]
second = [0, 4, 6, 7]
correctMatches = [6, 7]
foundMatches = similarities.findMatchingIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)

first = [1, 3, 5, 6, 7]
second = [0, 4, 6, 7, 8]
correctMatches = [6, 7]
foundMatches = similarities.findMatchingIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)

first = [1, 3, 5, 6, 7, 8]
second = [0, 4, 8, 9]
correctMatches = [8]
foundMatches = similarities.findMatchingIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)

first = [1, 3, 5, 6, 7, 8]
second = []
correctMatches = []
foundMatches = similarities.findMatchingIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)

# similarities.findMatchingAndUniqueIndices
first = [1, 3, 5, 6, 7]
second = [0, 4, 6, 7]
correctMatches = [6, 7]
correctFirst = [1, 3, 5]
correctSecond = [0, 4]
foundMatches, foundFirst, foundSecond = similarities.findMatchingAndUniqueIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)
assert correctFirst == foundFirst, "\nCorrect unique first: "+str(correctFirst)+"\nFound unique first: "+str(foundFirst)
assert correctSecond == foundSecond, "\nCorrect unique second: "+str(correctSecond)+"\nFound unique second: "+str(foundSecond)

first = [1, 3, 5, 6, 7, 8]
second = [0, 4, 6, 7]
correctMatches = [6, 7]
correctFirst = [1, 3, 5, 8]
correctSecond = [0, 4]
foundMatches, foundFirst, foundSecond = similarities.findMatchingAndUniqueIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)
assert correctFirst == foundFirst, "\nCorrect unique first: "+str(correctFirst)+"\nFound unique first: "+str(foundFirst)
assert correctSecond == foundSecond, "\nCorrect unique second: "+str(correctSecond)+"\nFound unique second: "+str(foundSecond)

first = [1, 3, 5, 6, 7]
second = [0, 4, 6, 7, 8]
correctMatches = [6, 7]
correctFirst = [1, 3, 5]
correctSecond = [0, 4, 8]
foundMatches, foundFirst, foundSecond = similarities.findMatchingAndUniqueIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)
assert correctFirst == foundFirst, "\nCorrect unique first: "+str(correctFirst)+"\nFound unique first: "+str(foundFirst)
assert correctSecond == foundSecond, "\nCorrect unique second: "+str(correctSecond)+"\nFound unique second: "+str(foundSecond)

first = [1, 3, 5, 6, 7, 8]
second = [0, 4, 8, 9]
correctMatches = [8]
correctFirst = [1, 3, 5, 6, 7]
correctSecond = [0, 4, 9]
foundMatches, foundFirst, foundSecond = similarities.findMatchingAndUniqueIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)
assert correctFirst == foundFirst, "\nCorrect unique first: "+str(correctFirst)+"\nFound unique first: "+str(foundFirst)
assert correctSecond == foundSecond, "\nCorrect unique second: "+str(correctSecond)+"\nFound unique second: "+str(foundSecond)

first = [1, 3, 5, 6, 7, 8]
second = []
correctMatches = []
correctFirst = [1, 3, 5, 6, 7, 8]
correctSecond = []
foundMatches, foundFirst, foundSecond = similarities.findMatchingAndUniqueIndices(first, second)
assert correctMatches == foundMatches, "\nCorrect matches: "+str(correctMatches)+"\nFound matches: "+str(foundMatches)
assert correctFirst == foundFirst, "\nCorrect unique first: "+str(correctFirst)+"\nFound unique first: "+str(foundFirst)
assert correctSecond == foundSecond, "\nCorrect unique second: "+str(correctSecond)+"\nFound unique second: "+str(foundSecond)

# similarities.cosineSim
firstFeatures = [1, 3, 5, 6, 7]
firstVector = [0, 2, 0, 3, 0, 1, 4, 5]
secondFeatures = [1, 2, 6, 7]
secondVector = [0, 2, 4, 0, 0, 0, 4, 5]
firstDoc = parsing.Doc(firstVector, firstFeatures)
secondDoc = parsing.Doc(secondVector, secondFeatures)
correctSim = spatial.distance.cosine(firstVector, secondVector)
sim = 1 - similarities.cosineSim(firstDoc, secondDoc)

assert correctSim == sim, "\nCorrect sim: "+str(correctSim)+" Sim: "+str(sim)

firstFeatures = [0, 1, 2]
firstVector = [1, 1, 1]
secondFeatures = [0, 1, 2]
secondVector = [1, 1, 1]
firstDoc = parsing.Doc(firstVector, firstFeatures)
secondDoc = parsing.Doc(secondVector, secondFeatures)
correctSim = spatial.distance.cosine(firstVector, secondVector)
sim = 1 - similarities.cosineSim(firstDoc, secondDoc)

assert correctSim == sim, "\nCorrect sim: "+str(correctSim)+" Sim: "+str(sim)

# Euclidean distance
firstFeatures = [1, 3, 5, 6, 7]
firstVector = [0, 2, 0, 3, 0, 1, 4, 5]
secondFeatures = [1, 2, 6, 7]
secondVector = [0, 2, 4, 0, 0, 0, 3, 5]
firstDoc = parsing.Doc(firstVector, firstFeatures)
secondDoc = parsing.Doc(secondVector, secondFeatures)
euclidDist = math.sqrt(similarities.euclideanDist(firstDoc, secondDoc))
euclidDistCorrect = np.linalg.norm(np.array(firstVector)-np.array(secondVector))
assert euclidDistCorrect == euclidDist, "\nCorrect Euclid: "+str(euclidDistCorrect)+" similarities.euclidDist: "+str(euclidDist)

print("Tests run successfully")
