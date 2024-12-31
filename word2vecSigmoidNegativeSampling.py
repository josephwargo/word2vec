import numpy as np

class word2vec(object):
    # constructor
    def __init__ (self, corpus, features, window, stepSize=.025, numNegSamples=5, minWordcount=2):
        """
        Constructor
        """
        self.features = features
        self.window = window
        self.stepSize = stepSize
        self.numNegSamples = numNegSamples
        self.labels = np.concatenate(([1], np.zeros(numNegSamples)))
        self.minWordcount = minWordcount
        self.corpus = self.initCorpus(corpus)
        self.allCorpusWords = [y for x in self.corpus for y in x]
        self.allCorpusWords = np.array(self.allCorpusWords)
        self.uniqueCorpusWords = list(set(self.allCorpusWords))
        self.uniqueCorpusWords.sort()
        # dictionary for index of each word
        self.word2ind = self.initWord2Ind()
        # random values to initiate embeddings
        np.random.seed(42)
        # self.allContext = np.random.normal(0, .001, (len(self.uniqueCorpusWords),features))
        self.allContext = np.zeros((len(self.uniqueCorpusWords),features))
        # self.allCenter = np.random.normal(0, .001, (len(self.uniqueCorpusWords),features))
        self.allCenter = np.random.uniform(0, 1, size=(len(self.uniqueCorpusWords),features))
    
    def initWord2Ind(self):
        """
        Creating dictionary that turns a string for a word into its embedding index
        """
        word2ind = {}
        for i in range(len(self.uniqueCorpusWords)):
            word2ind[self.uniqueCorpusWords[i]] = i
        return word2ind

    def initCorpus(self, corpus):
        flattenedCorpus = [y for x in corpus for y in x]
        uniqueWords, counts = np.unique(flattenedCorpus, return_counts=True)
        mask = counts <= self.minWordcount
        wordsToExclude = uniqueWords[mask]
        filteredCorpus = [[word for word in work if word not in wordsToExclude] for work in corpus]
        return filteredCorpus
    
    # singular version - to bugtest
    # def probability(self, contextEmbedding, centerEmbedding):
    #     """
    #     For a single center and context word pair:
    #         1. Takes the dot product of their respective embeddings
    #         2. Applies sigmoid
    #     """
    #     dotProduct = np.dot(contextEmbedding, centerEmbedding.T)
    #     # print(dotProduct)
    #     # print(dotProduct)
    #     normalizedDotProduct = dotProduct# - np.max(dotProduct)
    #     # if np.max(dotProduct>500):
    #         # print(np.min(dotProduct))
    #     probability = 1 / (1 + np.exp(-normalizedDotProduct))
    #     # print(probability)
    #     return probability

    def probability(self, contextEmbeddings, centerEmbedding):
        """
        For a single center and context word pair:
            1. Takes the dot product of their respective embeddings
            2. Applies sigmoid
        """
        dotProduct = np.dot(contextEmbeddings, centerEmbedding)
        # normalizedDotProduct = dotProduct - np.max(dotProduct)
        probability = 1 / (1 + np.exp(-dotProduct))
        return probability

    # singular version - to bugtest
    # def error(self, contextEmbedding, centerEmbedding):
    #     """
    #     tbd
    #     """
    #     prob = self.probability(contextEmbedding, centerEmbedding)
    #     # activation = self.labels - prob
    #     activation = 1 - prob
    #     error = activation * self.stepSize
    #     return error
    
    def error(self, contextEmbeddings, centerEmbedding):
        """
        tbd
        """
        prob = self.probability(contextEmbeddings, centerEmbedding)
        activation = self.labels - prob
        error = activation * self.stepSize
        return error
    
    # singular version - to bugtest
    # def updateEmbeddings(self, contextIndex, centerIndex):
    #     """
    #     tbd
    #     """
    #     contextEmbedding = self.allContext[contextIndex]
    #     centerEmbedding = self.allCenter[centerIndex]
    #     error = self.error(contextEmbedding, centerEmbedding)

    #     # updating context
    #     self.allContext[contextIndex] = contextEmbedding + centerEmbedding * error
    #     # self.allContext[contextIndex] = contextEmbedding + np.outer(centerEmbedding, error).T
    #     # updating center
    #     self.allCenter[centerIndex] = centerEmbedding + np.sum(contextEmbedding * error)
    #     # self.allCenter[centerIndex] = centerEmbedding + np.sum(contextEmbedding * error.reshape(1, len(error)).T)

    def updateEmbeddings(self, contextIndeces, centerIndex):
        """
        tbd
        """
        contextEmbeddings = self.allContext[contextIndeces]
        centerEmbedding = self.allCenter[centerIndex]
        error = self.error(contextEmbeddings, centerEmbedding)

        # updating context
        self.allContext[contextIndeces] = contextEmbeddings + np.outer(centerEmbedding, error).T
        # updating center
        self.allCenter[centerIndex] = centerEmbedding + np.sum(contextEmbeddings * error.reshape(1, len(error)).T)

    def negativeSampling(self, contextWord, centerWord):
        negativeSample = self.allCorpusWords[(self.allCorpusWords!=contextWord) & (self.allCorpusWords!=centerWord)]
        indeces = np.random.choice(len(negativeSample), size=self.numNegSamples, replace=False)
        negativeSample = negativeSample[indeces]
        return negativeSample

    def trainModel(self, epochs=1):
        """
        Update context and center word vectors while looping through the
        entire corpus
        """
        for i in range(epochs):
            for work in self.corpus:
                for wordIndex, centerWord in enumerate(work):
                    centerIndex = self.word2ind[centerWord]
                    for windowIndex in range(-self.window, self.window+1):
                        instanceIndex = wordIndex + windowIndex
                        if (instanceIndex >= 0) and (instanceIndex < len(work)):
                            if windowIndex != 0:
                                contextWord = work[instanceIndex]
                                negativeSample = self.negativeSampling(centerWord, contextWord)
                                # contextIndex = self.word2ind[contextWord]
                                contextIndeces = [self.word2ind[i] for i in negativeSample]
                                contextIndeces.insert(0,instanceIndex)
                                # self.updateEmbeddings(contextIndex, centerIndex)
                                self.updateEmbeddings(contextIndeces, centerIndex)

        # old
        # for i in range(epochs):
        #     for work in self.corpus:
        #         for wordIndex, word in enumerate(work):
        #             centerIndex = self.word2ind[word]
        #             for windowIndex in range(-self.window, self.window+1):
        #                 instanceIndex = wordIndex + windowIndex
        #                 if (instanceIndex >= 0) and (instanceIndex < len(work)):
        #                     if windowIndex != 0:
        #                         contextWord = work[instanceIndex]
        #                         contextIndex = self.word2ind[contextWord]
        #                         self.updateContext(contextIndex, centerIndex)
        #                         self.updateCenter(contextIndex, centerIndex)

    def predictWord(self, word):
        wordIndex = self.word2ind[word]
        centerEmbedding = self.allCenter[wordIndex]
        outputDistribution = np.dot(centerEmbedding, self.allContext.T)
        outputDistribution[wordIndex] = -10
        mostLikelyWordIndex = np.argmax(outputDistribution)
        mostLikelyWord = self.uniqueCorpusWords[mostLikelyWordIndex]
        return mostLikelyWord


if __name__ == "__main__":
    print("does this work?")