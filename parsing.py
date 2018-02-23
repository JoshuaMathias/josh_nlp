import math
import numpy as np

# Parsing and encoding utilities


def binToUTF8(string):
    return ('%x' % int(string, 2)).decode('hex').decode('utf-8')


# Returns a string of the rounded number, to the given number of digits after the decimal place.
def roundDecimals(num, numDecimalPlace):
    return '{0:.'+str(numDecimalPlace)+'f}'.format(num)


# Represent a document vector
class Doc:
    def __init__(self, vector, usedFeatures=[], category=None, index=None):
        self.vector = vector
        self.usedFeatures = usedFeatures
        self.category = category
        self.categoryI = -1
        self.index = index
        self.sumSquares = None
        self.score = 0
        self.predictedLabelI = -1

    # Assign this doc to the given category/class.
    def classify(self, newCategory):
        self.category = newCategory

    def setIndex(self, newIndex):
        self.index = newIndex

    def getIndex(self):
        return self.index

    def getCategory(self):
        return self.category

    def getFeature(self, featureIndex):
        return self.vector[featureIndex]

    def getVector(self):
        return self.vector

    def getUsedFeatures(self):
        return self.usedFeatures

    def calcSumOfSquares(self):
        self.sumSquares = 0
        for featI in self.usedFeatures:
            self.sumSquares += self.vector[featI]**2
            # print("sumSquares = "+str(self.sumSquares)+" after adding "+str(self.vector[featI]**2)+" for feature index "+str(featI))
        self.sumSquares = math.sqrt(self.sumSquares)

    def getSumSquares(self):
        if not self.sumSquares:
            self.calcSumOfSquares()
        return self.sumSquares

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score

    def flipScore(self):
        self.score = -self.score

    def extendNumFeatures(self, featureNum):
        numFeatures = len(self.vector)
        for i in range(featureNum-numFeatures):
            self.vector.append(0)

    def __str__(self):
        return "Document vector {\n\tindex: "+str(self.index)+",\n\tvector: "+str(self.vector)+\
               ",\n\tusedFeatures: "+str(self.usedFeatures)+",\n\tsumSquares: "+str(self.sumSquares)+\
               ",\n\tscore: "+str(self.score)+"\n}"

    def __lt__(self, x):
        return x.getScore() < self.getScore()


# Parse document vectors from format feat:num to a list of Doc
# Return (docs, featureNames, featureIndices)
def parseDocFeatureVectors(file_lines):
    # train_lines = open(training_data,'r').readlines()
    # Assign each feature an index.
    featureIndices = {}
    featureNames = []
    categoryIndices = {}
    categoryNames = []
    categoryNum = 0
    docs = []
    featureNum = 0
    docI = -1
    for line in file_lines:
        docI += 1
        features = line.split()
        if len(features) == 0:
            continue

        docCategory = features[0]
        if docCategory not in categoryIndices:
            categoryIndices[docCategory] = categoryNum
            categoryNum += 1
        docVector = []
        usedFeatures = []
        for featureI in range(1,len(features)):
            featureIndex = -1
            splitFeature = features[featureI].split(":")
            featureWord = splitFeature[0]

            # If the current feature isn't yet in featureIndices, add it and assign it the next index.
            if featureWord not in featureIndices:
                featureIndices[featureWord] = featureNum
                featureIndex = featureNum
                featureNum += 1
            else:
                featureIndex = featureIndices[featureWord]
            usedFeatures.append(featureIndex)

            # If the doc vector is less than the size of the index of the current feature,
            # then add to the doc vector until it reaches that index.
            if featureIndex >= len(docVector):
                for i in range(featureIndex-len(docVector)+1):
                    docVector.append(0)
            docVector[featureIndex] = int(splitFeature[1])
        newDoc = Doc(docVector, usedFeatures, docCategory, docI)
        docs.append(newDoc)
    numFeatures = len(featureIndices)
    numCategories = len(categoryIndices)
    # Extend each feature vector to the total number of features.
    for doc in docs:
        doc.extendNumFeatures(numFeatures)

    for i in range(numFeatures):
        featureNames.append("")

    for featureName in featureIndices:
        featureNames[featureIndices[featureName]] = featureName

    for i in range(numCategories):
        categoryNames.append("")

    for categoryName in categoryIndices:
        categoryNames[categoryIndices[categoryName]] = categoryName

    return docs, featureNames, featureIndices, categoryNames, categoryIndices


# Parse document vectors from format feat:num to a list of Doc
# Add only those features given. Ignore all others.
# Return docs
def parseDocVectorsSpecificFeatures(file_lines, featureIndices, categoryIndices, categoryNames):
    # train_lines = open(training_data,'r').readlines()

    docs = []
    docI = -1
    numFeatures = len(featureIndices)
    for line in file_lines:
        docI += 1

        features = line.split()
        if len(features) == 0:
            continue

        docCategory = features[0]
        # Initialize doc vector by size of given feature list.
        docVector = np.zeros(numFeatures)
        usedFeatures = []
        for featureI in range(1,len(features)):
            featureIndex = -1
            splitFeature = features[featureI].split(":")
            featureWord = splitFeature[0]

            # Only include feature if it's in featureIndices
            if featureWord in featureIndices:
                featureIndex = featureIndices[featureWord]
                usedFeatures.append(featureIndex)
                docVector[featureIndex] = int(splitFeature[1])
        newDoc = Doc(docVector, usedFeatures, docCategory, docI)
        newDoc.predictedLabelI = 0
        if docCategory in categoryIndices:
            newDoc.categoryI = categoryIndices[docCategory]
        else:
            newCatI = len(categoryIndices)
            categoryNames.append(docCategory)
            categoryIndices[docCategory] = newCatI
            newDoc.categoryI = newCatI
        docs.append(newDoc)

    return docs, categoryIndices, categoryNames


def lineToVectorBinary(line, featureIndices, categoryIndices):
    lineFeatures = line.split()
    docAnswer = lineFeatures[0]
    docAnswerI = categoryIndices[docAnswer]
    usedFeatures = []
    for lineFeature in lineFeatures[1:]:
        splitFeature = lineFeature.split(":")
        featureWord = splitFeature[0]
        if featureWord in featureIndices:
            usedFeatures.append(featureIndices[featureWord])
    return usedFeatures, docAnswer, docAnswerI


def wordFeaturesToVector(line, featureIndices, categoryIndices):
    lineFeatures = line.split()
    token = lineFeatures[0].split("-")[2]
    docAnswer = lineFeatures[1]
    docAnswerI = categoryIndices[docAnswer]
    usedFeatures = []
    for lineFeatureI in range(2,len(lineFeatures),2):
        featureWord = lineFeatures[lineFeatureI]
        if featureWord in featureIndices:
            usedFeatures.append(featureIndices[featureWord])
    return token, usedFeatures, docAnswer, docAnswerI


# Store value of feature as a tuple (featureIndex, value) in usedFeatures
def lineToVector(line, featureIndices, categoryIndices):
    lineFeatures = line.split()
    docAnswer = categoryIndices[lineFeatures[0]]
    usedFeatures = []
    for lineFeature in lineFeatures[1:]:
        splitFeature = lineFeature.split(":")
        featureWord = splitFeature[0]
        if featureWord in featureIndices:
            usedFeatures.append(featureIndices[featureWord], float(splitFeature[1]))
    return usedFeatures, docAnswer
