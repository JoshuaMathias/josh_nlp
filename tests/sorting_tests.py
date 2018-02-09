from unittest import TestCase
import unittest
from josh_nlp import counts
import numpy as np


class ParsingTestCase(TestCase):
    def setUp(self):
        return

    def test_(self):
        categories = []
        newCategory = counts.Category("cat1")
        newCategory.setScore(-1)
        categories.append(newCategory)
        newCategory = counts.Category("cat2")
        newCategory.setScore(4.5)
        categories.append(newCategory)
        newCategory = counts.Category("cat3")
        newCategory.setScore(0)
        categories.append(newCategory)
        newCategory = counts.Category("cat4")
        newCategory.setScore(2.1)
        categories.append(newCategory)
        highestCategory = 1
        # try:
        self.assertEqual(np.argmax(categories),highestCategory)
        # except AssertionError:
        #     print("argmax: "+str(np.argmax(categories)))
        catOrders = ["cat1", "cat3", "cat4", "cat2"]
        catCount = 0
        sortedCategories = sorted(categories)
        # print("sortedCategories 0: "+str(sortedCategories[0].getName()))
        for cat in sortedCategories:
            # try:
            self.assertEqual(sortedCategories[catCount].getName(),catOrders[catCount])

            catCount += 1


tester = ParsingTestCase()
unittest.main()
