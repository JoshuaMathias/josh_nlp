# Counting functions, such as for word, features, and classes.


class Category:
	def __init__(self, name, index=-1):
		self.name = name
		self.index = index
		self.docs = []
		self.featureCounts = []
		self.featureWeights = []
		self.prior = 0
		self.score = 0

	def getIndex(self):
		return self.index

	def addDoc(self, doc):
		self.docs.append(doc)
		# print("totalDocs: "+str(self.totalDocs))

	def setDocs(self,docs):
		self.docs = docs

	def getName(self):
		return self.name

	def addFeatureCount(self,featureI):
		if featureI >= len(self.featureCounts):
			for i in range(featureI-len(self.featureCounts)+1):
				self.featureCounts.append(0)
		self.featureCounts[featureI] += 1

	def extendNumFeatures(self,featureNum):
		for i in range(featureNum-len(self.featureCounts)+1):
			self.featureCounts.append(0)
		for doc in self.docs:
			for i in range(featureNum-len(doc)+1):
				doc.append(0)

	def setFeatureCounts(self,featureCounts):
		self.featureCounts = featureCounts

	def getFeatureCount(self,featureI):
		return self.featureCounts[featureI]

	def getFeatureWeight(self,featureI):
		return self.featureWeights[featureI]

	def getFeatureWeights(self):
		return self.featureWeights

	def setFeatureWeight(self, featureI, weight):
		if not self.featureWeights:
			self.featureWeights = []
		if featureI >= len(self.featureWeights):
			for i in range(featureI-len(self.featureWeights)+1):
				self.featureWeights.append(0)
		self.featureWeights[featureI] = weight
		# print("set feature weight "+str(featureI)+" "+str(weight))

	def extendNumWeights(self,featureNum):
		self.featureWeights = extendListLen(self.featureWeights, featureNum, 0)

	def subtractFeatureCounts(self, instances, features):
		for doc in instances:
			for feature in features:
				if doc[feature]:
					self.featureCounts[feature] -= 1

	def getNumDocs(self):
		return len(self.docs)

	def calcPrior(self, totalDocs, numCategories, classPriorDelta):
		self.prior = (classPriorDelta + len(self.docs)) / (numCategories * classPriorDelta + totalDocs)

	def getPrior(self):
		return self.prior

	def setPrior(self, prior):
		self.prior = prior

	def setScore(self, score):
		self.score = score

	def getScore(self):
		return self.score

	def __lt__(self, x):
		return self.getScore() > x.getScore()

	# def __gt__(self, x):
	# 	return self.getScore() > x.getScore()

	def __str__(self):
		if self.featureWeights is None:
			self.featureWeights = []
		return "name: "+str(self.name)+" score: "+str(self.score)+" prior: "+str(self.prior)+" num featureCounts: "+str(len(self.featureCounts))+" num featureWeights: "+str(len(self.featureWeights))


# Extend list to given length with given value
def extendListLen(inList, newLen, val):
	for i in range(newLen-len(inList)-1):
		inList.append(val)
	return inList


# Get feature and class/category counts.
# Create document vectors, each of the same size (of features), with True for present features and False for unpresent features.
# Return featureName, featureIndices, categoryNames, categoryIndices, docVectors
def countBinaryClassFeatures(train_lines):
	# train_lines = open(training_data,'r').readlines()
	# Assign each feature an index.
	featureIndices = {}
	featureNames = []
	categories = []
	categoryIndices = {}
	categoryNames = []
	docVectors = []
	featureNum = 0
	categoryNum = 0

	train_answers=[]
	for line in train_lines:
		features = line.split()
		if len(features) == 0:
			continue
		if features[0] not in categoryIndices:
			categoryIndices[features[0]] = categoryNum
			if categoryNum >= len(categories):
				for i in range(categoryNum-len(categories)+1):
					categories.append(None)
			categories[categoryNum] = Category(features[0])
			categoryNum += 1

		train_answers.append(categoryIndices[features[0]])
		category = categories[categoryIndices[features[0]]]
		# print("Adding count for category "+features[0])
		docVector = []
		for featureI in range(1,len(features)):
			featureIndex = -1
			splitFeature = features[featureI].split(":")
			featureWord = splitFeature[0]
			if featureWord not in featureIndices:
				featureIndices[featureWord] = featureNum
				featureIndex = featureNum
				featureNum += 1
			else:
				featureIndex = featureIndices[featureWord]
			category.addFeatureCount(featureIndex)
			if featureIndex >= len(docVector):
				for i in range(featureIndex-len(docVector)):
					docVector.append(0)
			else:
				docVector[featureIndex] = int(splitFeature[1])
		docVectors.append(docVector)
		category.addDoc(docVector)
	numFeatures = len(featureIndices)
	for category in categories:
		category.extendNumFeatures(numFeatures)
	numCategories = len(categoryIndices)

	for i in range(numFeatures):
		featureNames.append("")

	for featureName in featureIndices:
		featureNames[featureIndices[featureName]] = featureName

	for i in range(numCategories):
		categoryNames.append("")

	for categoryName in categoryIndices:
		categoryNames[categoryIndices[categoryName]] = categoryName

	return featureNames, featureIndices, categoryNames, categoryIndices, categories, docVectors
