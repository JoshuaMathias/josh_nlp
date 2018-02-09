from unittest import TestCase
import unittest
from josh_nlp import parsing


class ParsingTestCase(TestCase):
    def setUp(self):
        return

    def test_lineToVectorBinary(self):
        categoryIndices = {"test":0, "talk.politics.guns":1}
        featureIndices = {"a":0,"after":1,"test":2,"amend":3}
        line = "talk.politics.guns a:5 after:1 amend:1"
        usedFeatures, docAnswer = parsing.lineToVectorBinary(line, featureIndices, categoryIndices)
        self.assertTrue(usedFeatures == [0, 1, 3])
        self.assertTrue(docAnswer == 1)


tester = ParsingTestCase()
unittest.main()
