# Probability utilities/calculations

# Return a |proposals| x |givens| matrix of the conditional probabilities of proposals given givens, for each combination.
# Proposals is a list of Categories, each containing counts of features.
def conditionalProbs(categories, cond_smoothing_prob, distributionSize):
	if len(categories) == 0:
		return []
	conditionals = []
	numProposals = len(categories[0].featureCounts)
	for categoryI in range(len(categories)):
		features = categories[categoryI].featureCounts
		categoryCnt = categories[categoryI].getNumDocs()
		categoryConditionals = []
		if distributionSize == 2:
			for featureI in range(len(features)): 
				categoryConditionals.append((cond_smoothing_prob + features[featureI]) / (cond_smoothing_prob*distributionSize + categoryCnt))
		else:
			for featureI in range(len(features)): 
				categoryConditionals.append((cond_smoothing_prob + features[featureI]) / (cond_smoothing_prob*len(features) + categoryCnt))
		conditionals.append(categoryConditionals)
	return conditionals
