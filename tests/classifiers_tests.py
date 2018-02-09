from unittest import TestCase
import unittest
from josh_nlp import classifiers


class MaxEntTestCase(TestCase):
    def setUp(self):
        self.model_filename = "data/m1.txt"
        return

    # Test that the maxEnt has each category, which have featureWeights
    def test_loadModel(self):
        maxEnt = classifiers.MaxEnt()
        maxEnt.loadModel(self.model_filename)
        categories = maxEnt.categories
        self.assertEqual(len(categories),3)
        self.assertEqual(categories[0].getName(),"talk.politics.guns")
        # Check prior
        self.assertEqual('{0:.5f}'.format(categories[0].getPrior()),"-0.15835")
        # Ensure that feature indexing is consistent
        # foundLastFeat = False
        # for category in maxEnt.categories:
        #     print("category: "+str(category))
        #     self.assertEqual(maxEnt.numFeatures,len(category.getFeatureWeights()))
        #     if category.getFeatureWeight(maxEnt.numFeatures-1):
        #         foundLastFeat = True
        #     print(category)
        # self.assertTrue(foundLastFeat)
        featureIndices = maxEnt.featureIndices
        featI = 99
        featName = "soil"
        self.assertEqual(featI, featureIndices[featName])
        self.assertEqual('{0:.5f}'.format(categories[0].getFeatureWeight(featI)), "0.03068")
        self.assertEqual('{0:.5f}'.format(categories[1].getFeatureWeight(featI)), "0.00686")
        self.assertEqual('{0:.5f}'.format(categories[2].getFeatureWeight(featI)), "-0.03754")


unittest.main()
