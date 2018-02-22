import math
from josh_nlp import counts, probs, parsing, similarities
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from scipy import spatial


class NaiveBayes:
    def __init__(self, distribution_size=2):
        self.distributionSize = distribution_size
        self.categories = []
        self.conditionals = []
        self.unpresentConditionals = []
        self.numFeatures = 0
        self.numCategories = 0
        self.featureIndices = {}
        self.featureNames = []
        self.categoryIndices = {}
        self.categoryNames = []
        self.unpresentProbsProds = [] # for each category: product of 1 - P(w_k|c) over all possible words

    def train(self, train_lines, cond_smoothing_prob, class_prior_delta):
        self.featureNames, self.featureIndices, self.categoryNames, self.categoryIndices, self.categories, docVectors = counts.countBinaryClassFeatures(train_lines)
        self.numFeatures = len(self.featureIndices)
        self.numCategories = len(self.categoryIndices)
        numDocs = len(docVectors)

        for c in self.categories:
            c.calcPrior(numDocs, self.numCategories, class_prior_delta)

        self.conditionals = probs.conditionalProbs(self.categories, cond_smoothing_prob, self.distributionSize, docVectors)

    def prepareProbs(self):
        for con in self.conditionals:
            for concon in con:
                if concon < 0:
                    print("concon: "+str(concon))
        if self.distributionSize == 2:
            for i in range(self.numCategories):
                catList = []
                catConditionals = self.conditionals[i]
                for j in range(self.numFeatures):
                    condProb = catConditionals[j]
                    catList.append(math.log10(1-condProb))
                    self.conditionals[i][j] = math.log10(condProb)
                self.unpresentConditionals.append(catList)
            self.unpresentProbsProds = []
            for cI in range(self.numCategories):
                currProd = 0
                catConditionals = self.unpresentConditionals[cI]
                for featI in range(self.numFeatures):
                    currProd += catConditionals[featI]
                self.unpresentProbsProds.append(currProd)

    def test(self, test_lines):

        arrayNum = 0
        test_answers = []
        test_predicted = []
        sysStr = ""
        # print("num categories: "+str(len(self.categoryIndices)))
        # print("categoryIndices: "+str(self.categoryIndices))
        for line in test_lines:
            lineFeatures = line.split()
            currAnswer = self.categoryIndices[lineFeatures[0]]
            test_answers.append(currAnswer)
            vector = []
            vectorIndices = []
            for i in range(self.numFeatures):
                vector.append(0)
            for lineFeature in lineFeatures[1:]:
                splitFeature = lineFeature.split(":")
                featureWord = splitFeature[0]
                if featureWord in self.featureIndices:
                    vector[self.featureIndices[featureWord]] = int(splitFeature[1])
                    vectorIndices.append(self.featureIndices[featureWord])
            # Calculate probs for each category for this doc vector.
            categoryProbs = {}
            if self.distributionSize == 2:
                for cI in range(self.numCategories):
                    cTotal = self.categories[cI].prior
                    for featI in vectorIndices:
                        cTotal += self.conditionals[cI][featI] - self.unpresentConditionals[cI][featI]
                    cTotal += self.unpresentProbsProds[cI]
            else:
                # But you need to get the log probability of the feature in each c. Then, subtract the greatest log from all three logs. Then, the solution is 10^log pc / sum 10^log pc_i
                for cI in range(self.numCategories):
                    categoryProbs[cI] = self.categories[cI].prior
                for featI in range(self.numFeatures):
                    # Avoid underflow
                    highestProb = -float("inf")
                    featCats = []
                    for cI in range(self.numCategories):
                        featCats.append(self.conditionals[cI][featI])
                        if self.conditionals[cI][featI] > highestProb:
                            highestProb = self.conditionals[cI][featI]
                    highestCat = np.argmax(featCats)
                    featCats = featCats-highestCat
                    for cI in range(self.numCategories):
                        categoryProbs[cI] += featCats[cI]

            sysStr += "array:"+str(arrayNum)+" "+self.categoryNames[currAnswer]
            arrayNum += 1

            # Sort category probabilities:
            categoryProbs = [(k, categoryProbs[k]) for k in sorted(categoryProbs, key=categoryProbs.get, reverse=True)]
            print("category probs: "+str(categoryProbs))
            test_predicted.append(categoryProbs[0][0])
            for catProb in categoryProbs:
                sysStr += " "+self.categoryNames[catProb[0]]+" "+'{0:.5f}'.format(catProb[1]) # Print logs prob
            sysStr+="\n"
        return test_answers, test_predicted, sysStr

    def modelToString(self):
        modelStr = ""
        modelStr += "%%%%% prior prob P(c) %%%%%\n"
        for c in self.categories:
            modelStr += c.name+"\t"+str(c.prior)+"\t"+str(math.log10(c.prior))+"\n"
        modelStr += "%%%%% conditional prob P(f|c) %%%%%\n"
        for cI in range(self.numCategories):
            catName = self.categories[cI].name
            catConditionals = self.conditionals[cI]
            modelStr += "%%%%% conditional prob P(f|c) c="+catName+"%%%%%\n"
            for featureI in range(self.numFeatures):
                modelStr += self.featureNames[featureI]+"\t"+catName+"\t"+'{0:.5f}'.format(catConditionals[featureI])+"\t"+'{0:.5f}'.format(math.log10(catConditionals[featureI]))+"\n"
        return modelStr

    def accuracyToString(self, train_answers, train_predicted, test_answers, test_predicted):
        return accuracyToString(train_answers, train_predicted, test_answers, test_predicted, self.numCategories, self.categoryNames)


class KNN:
    def __init__(self):
        self.categories = []
        self.numFeatures = 0
        self.numCategories = 0
        self.featureIndices = {}
        self.featureNames = []
        self.categoryIndices = {}
        self.categoryNames = []
        self.docs = []
        self.preparedCosine = False

    def train(self, train_lines):
        self.docs, self.featureNames, self.featureIndices, self.categoryNames, self.categoryIndices = parsing.parseDocFeatureVectors(train_lines)
        self.numFeatures = len(self.featureIndices)
        self.numCategories = len(self.categoryIndices)

    # Prepare for more efficiency calculation of cosine similarities
    def prepareCosine(self):
        # Precalculate sum of squares
        for doc in self.docs:
            doc.calcSumOfSquares()

        self.preparedCosine = True

    # Classify test lines of document vectors and return test_answers, test_predicted, sysStr
    # similarityMeasure: An integer
    #   1 - Euclidean distance
    #   2 - Cosine similarity
    def test(self, test_lines, similarityMeasure, K):
        arrayNum = 0
        test_answers = []
        test_predicted = []
        sysStr = ""
        isCosine = False

        if similarityMeasure == 2:
            isCosine = True
            self.prepareCosine()

        for line in test_lines:
            # docScores = []
            lineFeatures = line.split()
            if lineFeatures[0] in self.categoryIndices:
                currAnswer = self.categoryIndices[lineFeatures[0]]
            else:
                currAnswer = self.numCategories+1
            test_answers.append(currAnswer)
            vector = []
            vectorIndices = []
            for i in range(self.numFeatures):
                vector.append(0)
            for lineFeature in lineFeatures[1:]:
                splitFeature = lineFeature.split(":")
                featureWord = splitFeature[0]
                if featureWord in self.featureIndices:
                    vector[self.featureIndices[featureWord]] = int(splitFeature[1])
                    vectorIndices.append(self.featureIndices[featureWord])

            testDoc = parsing.Doc(vector, usedFeatures=vectorIndices)
            # Find K most similar document vectors
            for doc in self.docs:
                simScore = 0
                if isCosine:
                    # simScore = similarities.cosineSim(testDoc,doc)
                    simScore = spatial.distance.cosine(testDoc.getVector(), doc.getVector())
                else:
                    simScore = similarities.euclideanDist(testDoc, doc)
                    # np.linalg.norm(np.array(testDoc.getVector())-np.array(doc.getVector()))
                doc.setScore(simScore)
                # docScores.append(simScore)

            # For Euclidean: Smallest values
            # For Cosine: Greatest values (because we don't do 1 - value)

            if not isCosine:
                for doc in self.docs:
                    doc.flipScore()
            kBest = np.argpartition(self.docs, range(K))
            kBest = kBest[:K]

            categoryProbs = {}
            for cI in range(self.numCategories):
                categoryProbs[cI] = 0

            for voterI in kBest:
                voter = self.docs[voterI]
                print("best doc score: "+str(voter.getScore()))
                categoryProbs[self.categoryIndices[voter.getCategory()]] += 1
            print("category votes: "+str(categoryProbs))
            sysStr += "array:"+str(arrayNum)+" "+self.categoryNames[currAnswer]
            arrayNum += 1

            # Sort category probabilities:
            categoryProbs = [(k, categoryProbs[k]) for k in sorted(categoryProbs, key=categoryProbs.get, reverse=True)]
            test_predicted.append(categoryProbs[0][0])
            # print("test predicted: "+str(categoryProbs[0][0]))
            for catProb in categoryProbs:
                sysStr += " "+self.categoryNames[catProb[0]]+" "+str(catProb[1])
            sysStr+="\n"
            # break
        return test_answers, test_predicted, sysStr

    def accuracyToString(self, train_answers, train_predicted, test_answers, test_predicted):
        return accuracyToString(train_answers, train_predicted, test_answers, test_predicted, self.numCategories, self.categoryNames)


# Calculate confusion matrices and accuracy scores for training and testing, and return a string showing results.
def accuracyToString(train_answers, train_predicted, test_answers, test_predicted, numCategories, categoryNames):
    labels = [i for i in range(numCategories)]
    outStr = "Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n\t"
    trainMatrix = confusion_matrix(train_answers, train_predicted, labels=labels)
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
    outStr+="\n"
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
        for c2 in range(numCategories):
            outStr+=str(trainMatrix[c][c2])+" "
        outStr += "\n"
    outStr+="\n"
    outStr += "Training accuracy="+'{0:.5f}'.format(accuracy_score(train_predicted,train_answers))+"\n\n\n"

    outStr += "Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n\t"
    testMatrix = confusion_matrix(test_answers, test_predicted, labels=labels)
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
    outStr += "\n"
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
        for c2 in range(numCategories):
            outStr+=str(testMatrix[c][c2])+" "
        outStr += "\n"
    outStr+="\n"

    outStr += "Test accuracy="+'{0:.5f}'.format(accuracy_score(test_predicted,test_answers))
    return outStr


# Calculate confusion matrices and accuracy scores for testing, and return a string showing results.
def testAccuracyToString(test_answers, test_predicted, numCategories, categoryNames):
    outStr = ""
    # print("test_answers: "+str(test_answers)+"\ntest_predicted: "+str(test_predicted))
    outStr += "Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n\t"
    labels = [i for i in range(numCategories)]
    testMatrix = confusion_matrix(test_answers, test_predicted, labels=labels)
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+"\t"
    outStr += "\n"
    for c in range(len(testMatrix)):
        outStr+=str(categoryNames[c])+"\t"
        for c2 in range(len(testMatrix[c])):
            outStr+=str(testMatrix[c][c2])+"\t"
        outStr += "\n"
    outStr+="\n"

    outStr += "Test accuracy="+'{0:.5f}'.format(accuracy_score(test_predicted,test_answers))
    return outStr


class BeamNode:
    def __init__(self, prevNode, nextClass): # nextClass is a Category object
        if prevNode:
            self.classes = list(prevNode.getClasses())
            self.classes.append((nextClass.getName(),nextClass.getIndex(),nextClass.getScore()))
            self.prob = prevNode.getProb() + nextClass.getScore()
            self.prevT = None
            self.prevTwoTags = None
        else:
            self.classes = [(nextClass.getName(),nextClass.getIndex(),nextClass.getScore())]
            self.classes.append((nextClass.getName(),nextClass.getIndex(),nextClass.getScore()))
            self.prob = nextClass.getScore()
            self.prevT = None
            self.prevTwoTags = None

    def getClasses(self):
        return self.classes

    def getProb(self):
        return self.prob

    def getPrevT(self):
        if not self.prevT:
            if len(self.classes):
                self.prevT = "prevT="+self.classes[-1][0]
            else:
                self.prevT = "prevT=BOS"
        return self.prevT

    def getPrevTwoTags(self):
        if not self.prevTwoTags:
            featStr = "prevTwoTags="
            if len(self.classes) > 1:
                featStr += self.classes[-2][0]+"+"+self.classes[-1][0]
            else:
                featStr += "BOS+"
                if len(self.classes):
                    featStr += self.classes[-1][0]
                else:
                    featStr += "BOS"
            self.prevTwoTags = featStr
        return self.prevTwoTags

    def __lt__(self, x):
        return self.getProb() > x.getProb()

    def __str__(self):
        return "classes: "+str(self.classes)+"\nprob: "+str(self.prob)


class MaxEnt:
    def __init__(self):
        self.categories = []
        self.numFeatures = 0
        self.numCategories = 0
        self.featureIndices = {}
        self.featureNames = []
        self.categoryIndices = {}
        self.categoryNames = []
        self.docs = []
        self.preparedCosine = False
        self.featureWeights = []

    def loadModel(self, model_filename):
        modelFile = open(model_filename, 'r')
        modelLines = modelFile.readlines()
        modelFile.close()
        currCategory = None
        for line in modelLines:
            splitLine = line.strip().split()

            if line.startswith("FEATURES FOR CLASS"):
                # if currCategory:
                    # print("currCategory weights len: "+str(len(currCategory.getFeatureWeights())))
                    # print("currCategory: "+str(currCategory))
                    # self.categories.append(currCategory)

                currCategory = counts.Category(splitLine[-1], index=self.numCategories)

                self.categoryIndices[currCategory.getName()] = self.numCategories
                self.categoryNames.append(currCategory.getName())
                self.numCategories += 1
                self.categories.append(currCategory)
            elif splitLine[0] == "<default>":
                currCategory.setPrior(float(splitLine[-1]))
            else:
                splitLine = line.split()
                feature = splitLine[0]
                weight = float(splitLine[1])
                featI = -1
                if feature not in self.featureIndices:
                    self.featureIndices[feature] = self.numFeatures
                    featI = self.numFeatures
                    self.numFeatures += 1
                else:
                    featI = self.featureIndices[feature]
                # print("featI: "+str(featI)+" feature: "+feature+" weight: "+str(weight))
                currCategory.setFeatureWeight(featI, weight)
        # print("numFeatures: "+str(self.numFeatures))
        for cat in self.categories:
            cat.extendNumWeights(self.numFeatures)
            # print("new category: "+str(cat))

    def test(self, test_lines):
        test_predicted = []
        test_answers = []
        sysStr = ""
        arrayNum = 0

        for line in test_lines:
            usedFeatures, currAnswer, currAnswerI = parsing.lineToVectorBinary(line, self.featureIndices, self.categoryIndices)
            Z = 0
            for category in self.categories:
                featSum = 0
                for featI in usedFeatures:
                    featSum += category.getFeatureWeight(featI)
                    numerator = math.exp(featSum)
                    category.setScore(numerator)
                    Z += numerator

            for category in self.categories:
                category.setScore(category.getScore() / Z)
            sortedCategories = sorted(self.categories, reverse=True)
            test_predicted.append(sortedCategories[0].getIndex())
            test_answers.append(currAnswerI)
            sysStr += "array:"+str(arrayNum)+" "+currAnswer
            arrayNum += 1
            for category in sortedCategories:
                sysStr += " "+category.getName()+" "+str(category.getScore())
            sysStr += "\n"
            # break
        return test_answers, test_predicted, sysStr

    def testBeam(self, test_lines, boundary_lines, topN, topK, beam_size):
        test_predicted = []
        test_answers = []
        sysStr = ""
        currSentI = 0
        currBoundary = int(boundary_lines[currSentI])
        currWordI = 0
        currAnswerI = None
        currAnswer = None
        topPaths = []
        paths = []
        paths.append(BeamNode(None,counts.Category("BOS")))
        sentTokens = []

        for line in test_lines:
            token, usedFeatures, currAnswer, currAnswerI = parsing.wordFeaturesToVector(line, self.featureIndices, self.categoryIndices)
            test_answers.append(currAnswerI)
            sentTokens.append(token)

            nextPaths = []
            # print("paths before while loop: "+str(len(paths)))
            while len(paths):
                currPath = paths.pop()
                # print("Used features: "+str(usedFeatures)+" currAnswer: "+str(currAnswer)+" currAnswerI: "+str(currAnswerI))
                # Add prevT and prevTwoTags to calculation, for this specific path.
                prevT = currPath.getPrevT()
                prev2 = currPath.getPrevTwoTags()
                # print("prevT: "+prevT+" prevTwoTags: "+prev2)

                Z = 0
                for category in self.categories:
                    # Sum lamdas for all present features
                    featSum = category.getPrior()
                    for featI in usedFeatures:
                        featSum += category.getFeatureWeight(featI)
                    # Include prevT
                    if prevT in self.featureIndices:
                        featSum += category.getFeatureWeight(self.featureIndices[prevT])
                    # Include prevTwoTags
                    if prev2 in self.featureIndices:
                        featSum += category.getFeatureWeight(self.featureIndices[prev2])

                    featSum = math.exp(featSum)
                    # print("category: "+str(category)+" numerator: "+str(featSum))
                    category.setScore(featSum)
                    Z += featSum

                for category in self.categories:
                    # print("category score: "+str(category.getScore()))
                    category.setScore(math.log(category.getScore() / Z))

                if len(self.categories) > topN:
                    topCategories = np.argpartition(self.categories, range(topN))
                    topCategories = topCategories[:topN]
                for categoryI in topCategories:
                    nextPaths.append(BeamNode(currPath, self.categories[categoryI]))

            # print("next paths: "+str(len(nextPaths)))
            topPaths = [i for i in range(len(nextPaths))] # Use all paths if there are not more than topK.
            if len(nextPaths) > topK: # Prune paths
                topPaths = np.argpartition(nextPaths, range(topK))
                # print("topPaths before split: "+str(len(topPaths)))
                topPaths = topPaths[:topK]
            # print("topPaths: "+str(len(topPaths)))
            prunedPaths = []
            if len(topPaths):
                max_prob = nextPaths[topPaths[0]].getProb()
                # print("max prob: "+str(max_prob))
                for pathI in topPaths:
                    # print("Possible prob to add: "+str(nextPaths[pathI].getProb())+" plus beam: "+str(nextPaths[pathI].getProb() + beam_size))
                    if nextPaths[pathI].getProb() + beam_size >= max_prob: # Probabilities are minused, for argpartition
                        prunedPaths.append(nextPaths[pathI])
                    else:
                        break # It only gets worse at this point (top paths are sorted).
            paths = prunedPaths
            currWordI += 1
            if not currWordI < currBoundary: # Are we past the last token of the last sentence?
                # print("currWordI: "+str(currWordI)+" currBoundary: "+str(currBoundary))
                # Choose the best final path, and output
                if len(paths):
                    classes = paths[0].getClasses()
                    classI = 0
                    for predicted in classes: # Output answer and prediction for each token in best path.
                        if predicted[0] == "BOS": # Don't output BOS tags (doesn't correspond with a token)
                            continue
                        test_predicted.append(predicted[1])
                        sysStr += str(currSentI)+"-"+str(classI)+"-"+sentTokens[classI]+" "+self.categoryNames[test_answers[-currBoundary+classI]]+" "+predicted[0]+" "+str(10**predicted[2])
                        classI += 1
                        sysStr += "\n"

                # Start the next sentence
                currWordI = 0
                currSentI += 1
                if currSentI < len(boundary_lines):
                    currBoundary = int(boundary_lines[currSentI])
                    paths = []
                    paths.append(BeamNode(None,counts.Category("BOS")))
                    sentTokens = []
                else:
                    break

        return test_answers, test_predicted, sysStr

    # Return catProbs
    def getCondProbs(self, usedFeatures):
        catProbs = []
        Z = 0
        for category in self.categories:
            featSum = 0
            for featI in usedFeatures:
                featSum += category.getFeatureWeight(featI)
                numerator = math.exp(featSum)
                catProbs.append(numerator)
                Z += numerator
        for categoryI in range(len(self.categories)):
            catProbs[categoryI] = (catProbs[categoryI] / Z)
        return catProbs

    def accuracyToString(self, test_answers, test_predicted):
        return testAccuracyToString(test_answers, test_predicted, self.numCategories, self.categoryNames)


# Calculate empirical expectation
def calcEmpExpectation(train_lines):
    featureNames, featureIndices, categoryNames, categoryIndices, categories, docVectors = counts.countBinaryClassFeatures(train_lines)
    N = len(docVectors)

    sortedFeatures = []
    sortedNames = sorted(featureNames)
    for featName in sortedNames:
        featI = featureIndices[featName]
        sortedFeatures.append((featI,featName))

    sysStr = ""
    for category in categories:
        for feat in sortedFeatures:
            featureCount = category.getFeatureCount(feat[0])
            expect = featureCount / N
            if not expect:
                continue
            sysStr += category.getName()+" "+feat[1]+" "+'{0:.5f}'.format(expect)+" "+str(featureCount)+"\n"
    return sysStr


# Calculate model expectation
def calcModelExpectation(train_lines, model_filename):
    featureNames, featureIndices, categoryNames, categoryIndices, categories, docVectors = counts.countBinaryClassFeatures(train_lines)
    N = len(docVectors)

    maxEnt = None
    if model_filename:
        maxEnt = MaxEnt()
        maxEnt.loadModel(model_filename)

    sortedFeatures = []
    sortedNames = sorted(featureNames)
    for featName in sortedNames:
        featI = featureIndices[featName]
        sortedFeatures.append((featI,featName))

    model_expect = []
    if maxEnt:
        for catI in range(len(categories)):
            catList = []
            for featI in range(len(featureNames)):
                catList.append(0)
            model_expect.append(catList)
        for line in train_lines:
            usedFeatures, currAnswer, currAnswerI = parsing.lineToVectorBinary(line, featureIndices, categoryIndices)
            catProbs = maxEnt.getCondProbs(usedFeatures)
            for featureI in usedFeatures:
                for categoryI in range(len(categoryNames)):
                    model_expect[categoryI][featureI] += catProbs[categoryI]

    sysStr = ""
    defaultProb = 1 / len(categories)
    categoryI = 0
    for category in categories:
        featI = 0
        for feat in sortedFeatures:
            featureCount = category.getFeatureCount(feat[0])
            expect = featureCount / N
            if maxEnt:
                expect *= model_expect[categoryI][featI]
            else:
                expect *= defaultProb
            if not expect:
                continue
            sysStr += category.getName()+" "+feat[1]+" "+'{0:.5f}'.format(expect)+" "+str(featureCount)+"\n"
            featI += 1
        categoryI += 1
    return sysStr


class TBL:
    def __init__(self):
        self.categories = []
        self.numFeatures = 0
        self.numCategories = 0
        self.featureIndices = {}
        self.featureNames = []
        self.categoryIndices = {}
        self.categoryNames = []
        self.docs = []
        self.netGains = []
        self.defaultLabelI = 0

    # def __loadData(self, train_lines):

    def train(self, train_lines, min_gain):
        # loadData(train_lines)
        print("train")
        # Start with the label of the first training instance
        self.docs, self.featureNames, self.featureIndices, self.categoryNames, self.categoryIndices = parsing.parseDocFeatureVectors(train_lines)
        self.numFeatures = len(self.featureNames)
        self.numCategories = len(self.categoryNames)
        for i in range(self.numFeatures):
            featList = []
            for j in range(self.numCategories):
                catList = []
                for k in range(self.numCategories):
                    catList.append(0)
                featList.append(catList)
            self.netGains.append(featList)
        self.defaultLabelI = self.categoryIndices[self.docs[0].getCategory()]
        nonDefaultLabels = []
        for categoryI in range(self.numCategories):
            if categoryI == self.defaultLabelI:
                continue
            nonDefaultLabels.append(categoryI)

        # Calculate net gains of all transformations by updating feature counts for each training instance.
        # TODO Loop len(transformations)
        # TODO keep track of current labels for docs
        
        for doc in self.docs:
            for featureI in range(self.numFeatures):
                for label in nonDefaultLabels:
                    docCatI = self.categoryIndices[doc.getCategory()]
                    if self.defaultLabelI == docCatI:
                        self.netGains[featureI][label] += 1
                    else:
                        self.netGains[featureI][label] -= 1

    def test(self, test_lines):
        print("test")

    def saveModel(self, model_filename):
        print("saveModel")
        modelStr = ""
        defaultLabel = self.categoryNames[self.defaultLabelI]
        for featI in range(self.numFeatures):
            for catI in range(1,self.numCategories):
                modelStr += self.featureNames[featI]+" "+defaultLabel+" "+self.categoryNames[catI]+" "+str(self.netGains[featI][catI])+"\n"

        modelFile = open(model_filename, 'w')
        modelFile.write(modelStr)

    def loadModel(self, model_filename):
        print("loadModel")
