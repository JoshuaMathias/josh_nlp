class Category:
	def __init__(self, name):
		self.name = name
		self.docs = []
		self.featuresCounts = []

	def addDoc(self, doc):
		self.docs.append(doc)
		# print("totalDocs: "+str(self.totalDocs))

	def setDocs(self,docs):
		self.docs = docs

	def addFeatureCount(self,featureI):
		if featureI >= len(self.featuresCounts):
			for i in range(featureI-len(self.featuresCounts)+1):
				self.featuresCounts.append(0)
		self.featuresCounts[featureI] += 1

	def extendNumFeatures(self,featureNum):
		for i in range(featureNum-len(self.featuresCounts)+1):
			self.featuresCounts.append(0)
		for doc in self.docs:
			for i in range(featureNum-len(doc)+1):
				doc.append(False)

	def setFeaturesCounts(self,featureCounts):
		self.featuresCounts = featureCounts

	def getFeatureCount(self,featureI):
		return self.featuresCounts[featureI]

	def subtractFeatureCounts(self, instances, features):
		for doc in instances:
			for feature in features:
				if doc[feature]:
					self.featuresCounts[feature] -= 1

	def getNumDocs(self):
		return len(self.docs)


# Get feature and class/category counts.
# Create document vectors, each of the same size (of features), with True for present features and False for unpresent features.
# Return featureName, featureIndices, categoryNames, categoryIndices, docVectors
def countBinaryClassFeatures(training_data):
	train_lines = open(training_data,'r').readlines()
	# Assign each feature an index.
	featureIndices = {}
	featureNames = []
	# Get feature frequencies for each class
	classVectors = {}
	classCounts = {}
	categories = []
	categoryIndices = {}
	categoryNames = []
	docVectors = []
	featureNum = 0
	categoryNum = 0

	train_answers=[]
	for line in train_lines:
		features = line.split()
		if features[0] not in categoryIndices:
			categoryIndices[features[0]] = categoryNum
			if categoryNum >= len(categories):
				for i in range(categoryNum-len(categories)+1):
					categories.append(None)
			categories[categoryNum] = Category(features[0])
			categoryNum += 1
		
		train_answers.append(categoryIndices[features[0]])
		category = categories[categoryIndices[features[0]]]
		print("Adding count for category "+features[0])
		docVector = []
		for featureI in range(1,len(features)):
			featureIndex = -1
			featureWord = features[featureI].split(":")[0]
			if featureWord not in featureIndices:
				featureIndices[featureWord] = featureNum
				featureIndex = featureNum
				featureNum += 1
			else:
				featureIndex = featureIndices[featureWord]
			category.addFeatureCount(featureIndex)
			if featureIndex >= len(docVector):
				for i in range(featureIndex-len(docVector)):
					docVector.append(False)
				docVector.append(True)
			else:
				docVector[featureIndex] = True
		docVectors.append(docVector)
		category.addDoc(docVector)
	numFeatures = len(featureIndices)
	for category in categories:
		category.extendNumFeatures(numFeatures)
	numCategories = len(categoryIndices)
	return featureName, featureIndices, categoryNames, categoryIndices, docVectors