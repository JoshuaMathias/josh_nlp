# Probability utilities/calculations

# Return a |proposals| x |givens| matrix of the conditional probabilities of proposals given givens, for each combination.
# Proposals is a list of Categories, each containing counts of features.
def conditionalProbs(categories, cond_smoothing_prob, distributionSize, docs = None):
	if len(categories) == 0:
		return []
	conditionals = []
	numProposals = len(categories[0].featureCounts)
	featCategoryCounts = []
	categoryTotals = []
	if distributionSize != 2:
		for c in range(len(categories)):
			categoryCounts = []
			categoryTotals.append(0)
			for c in range(len(docs[0])):
				categoryCounts.append(0)
			featCategoryCounts.append(categoryCounts)
		for doc in docs:
			for cj in range(len(categories)):
				for featI in range(len(doc)):
					feat = doc[featI]
					if feat:
						featCategoryCounts[cj][featI] += feat
						categoryTotals[cj] += feat
	for categoryI in range(len(categories)):
		features = categories[categoryI].featureCounts
		categoryConditionals = []
		if distributionSize == 2:
			for featureI in range(len(features)):
				categoryCnt = categories[categoryI].getNumDocs()
				categoryConditionals.append((cond_smoothing_prob + features[featureI]) / (cond_smoothing_prob*distributionSize + categoryCnt))
		else:
			for featureI in range(len(features)):
				categoryConditionals.append((cond_smoothing_prob + featCategoryCounts[categoryI][featureI]) / (cond_smoothing_prob*len(features) + categoryTotals[categoryI]))
		conditionals.append(categoryConditionals)
	return conditionals