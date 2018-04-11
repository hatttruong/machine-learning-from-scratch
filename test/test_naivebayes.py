#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Testcases for text_preprocessing module

"""
import logging
import sys
import unittest
sys.path.append('../')

from supervised.naivebayes import MultinomialNB

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TestNaiveBayesModel(unittest.TestCase):

    """
    Test cases for Naive Bayes model

    Attributes:
        categories (list): Description
        test_documents (list): Description
        train_documents (TYPE): Description

    """
    # category "1" means "China", 0 means "not China"
    categories = [1, 0]
    train_documents = [('Chinese Beijing Chinese', 'yes'),
                       ('Chinese Chinese Shanghai', 'yes'),
                       ('Chinese Macao', 'yes'),
                       ('Tokyo Japan Chinese', 'no')]
    test_documents = ['Chinese Chinese Chinese Tokyo Japan']

    def test_train(self):
        """Summary
        """
        nb_clf = MultinomialNB()
        nb_clf.train([d[0] for d in self.train_documents],
                     [d[1] for d in self.train_documents])
        self.assertEqual(nb_clf.num_documents, len(self.train_documents))
        self.assertEqual(nb_clf.label2index, {'no': 0, 'yes': 1})
        self.assertEqual(nb_clf.prior, [0.25, 0.75])
        self.assertEqual(nb_clf.word2index,
                         {
                             'chinese': 0, 'beijing': 1, 'shanghai': 2,
                             'macao': 3, 'tokyo': 4, 'japan': 5,
                         })
        # self.assertEqual(nb_clf.condprob,
        #                  [
        #                      {0: 1, 4: 1, 5: 1},
        #                      {0: 5, 1: 2, 2: 1, 3: 1},
        #                  ])
        # label "no", word=chinese
        self.assertEqual(nb_clf.condprob[0][0], (1 + 1) * 1.0 / (3 + 6))
        # label "no", word=beijing
        self.assertEqual(nb_clf.condprob[0][1], (0. + 1) * 1.0 / (3 + 6))
        # label "yes", word=chinese
        self.assertEqual(nb_clf.condprob[1][0], (5 + 1) * 1.0 / (8 + 6))
        # label "yes", word=japan
        self.assertEqual(nb_clf.condprob[1][4], (0. + 1) * 1.0 / (8 + 6))

    def test_predict(self):
        """Summary
        """
        nb_clf = MultinomialNB()
        nb_clf.train([d[0] for d in self.train_documents],
                     [d[1] for d in self.train_documents])
        label = nb_clf.predict(self.test_documents[0])
        self.assertEqual(label, 'yes')


if __name__ == '__main__':
    unittest.main()
