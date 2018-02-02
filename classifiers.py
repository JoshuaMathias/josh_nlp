import math
from josh_nlp import counts, probs, parsing, similarities
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np


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
        for con in self.conditionals:
            for concon in con:
                if concon < 0:
                    print("concon: "+str(concon))

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

    def test(self, test_data):

        arrayNum = 0
        test_lines = open(test_data,'r').readlines()
        test_answers = []
        test_predicted = []
        sysStr = ""
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
                    simScore = similarities.cosineSim(testDoc,doc)
                else:
                    simScore = similarities.euclideanDist(testDoc, doc)
                doc.setScore(simScore)
                # docScores.append(simScore)

            # For Euclidean: Smallest values
            # For Cosine: Greatest values

            if not isCosine:
                for doc in self.docs:
                    doc.flipScore()
            kBest = np.argpartition(self.docs, K)
            kBest = kBest[:K]

            categoryProbs = {}
            for cI in range(self.numCategories):
                categoryProbs[cI] = 0

            for voterI in kBest:
                voter = self.docs[voterI]
                # print("best doc score: "+str(voter.getScore()))
                categoryProbs[self.categoryIndices[voter.getCategory()]] += 1
            # print("category votes: "+str(categoryProbs))
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


# Calculate confusion matrices and accuracy scores and training and testing, and return a string showing results.
def accuracyToString(train_answers, train_predicted, test_answers, test_predicted, numCategories, categoryNames):
    outStr = "Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n\t"
    trainMatrix = confusion_matrix(train_answers, train_predicted)
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
    testMatrix = confusion_matrix(test_answers, test_predicted)
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
    for c in range(numCategories):
        outStr+=str(categoryNames[c])+" "
        for c2 in range(numCategories):
            outStr+=str(testMatrix[c][c2])+" "
        outStr += "\n"
    outStr+="\n"

    outStr += "Test accuracy="+'{0:.5f}'.format(accuracy_score(test_predicted,test_answers))
    return outStr
