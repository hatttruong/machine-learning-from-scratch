from collections import Counter
import logging
import numpy as np
import operator

from util import helper

logger = logging.getLogger(__name__)


class MultinomialNB(object):

    """
    A simple Multinomial Naive Bayes for text classification

    Attributes:
        condprob (TYPE): Description
        prior (TYPE): Description
        word2index (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        self.word2index = dict()
        self.prior = list()
        self.condprob = None
        self.num_documents = 0
        self.label2index = dict()

    def reset(self):
        """Summary
        """
        self.word2index = dict()
        self.prior = list()
        self.condprob = None
        self.num_documents = 0
        self.label2index = dict()

    def train(self, documents, labels):
        """Summary

        Args:
            documents (TYPE): Description
            labels (TYPE): Description
        """
        self.reset()

        self.num_documents = len(documents)

        # calculate prior
        label_counter = Counter(labels)
        for l in sorted(label_counter.keys()):
            self.label2index[l] = len(self.prior)
            self.prior.append(label_counter[l] * 1.0 / self.num_documents)

        # calculate conditional probability
        wordcount_per_label = [dict() for i in range(len(self.label2index))]
        logger.debug('size of wordcount_per_label: %s',
                     len(wordcount_per_label))
        for doc, label in zip(documents, labels):
            label_idx = self.label2index[label]
            logger.debug('documents="%s", label=%s, label_index=%s',
                         doc, label, label_idx)

            words_per_doc = helper.extract_words(doc)
            for w in words_per_doc:
                if w not in self.word2index.keys():
                    self.word2index[w] = len(self.word2index)
                w_idx = self.word2index[w]
                if w_idx in wordcount_per_label[label_idx].keys():
                    wordcount_per_label[label_idx][w_idx] += 1
                else:
                    wordcount_per_label[label_idx][w_idx] = 1
            logger.debug('wordcount_per_label: %s', wordcount_per_label)

        self.condprob = [dict() for i in range(len(self.label2index))]
        for label_idx in self.label2index.values():
            nb_words = sum(wordcount_per_label[label_idx].values()) + \
                len(self.word2index)

            for w_idx in self.word2index.values():
                wc = 1
                if w_idx in wordcount_per_label[label_idx].keys():
                    wc += wordcount_per_label[label_idx][w_idx]
                p = wc * 1.0 / nb_words
                self.condprob[label_idx][w_idx] = p

        logger.info('condprob: %s', self.condprob)

    def predict(self, document):
        """Summary

        Args:
            documents (TYPE): Description
        """
        words_per_doc = helper.extract_words(document)
        scores = dict()
        for l_name, l_idx in self.label2index.items():
            scores[l_name] = np.log(self.prior[l_idx])
            logger.debug('prior[%s]= %s', l_name, self.prior[l_idx])
            for w in words_per_doc:
                logger.debug('condprob word[%s]= %s', w,
                             self.condprob[l_idx][self.word2index[w]])
                scores[l_name] += np.log(
                    self.condprob[l_idx][self.word2index[w]])

        logger.info('log(scores): %s', scores)
        logger.info('scores (P): %s', [(l, np.exp(s))
                                       for l, s in scores.items()])
        return max(scores.items(), key=operator.itemgetter(1))[0]
