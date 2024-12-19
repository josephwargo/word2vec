import numpy as np

class word2vec(object):
    # constructor
    def __init__ (self, corpus, features, window, stepSize=.025):
        """
        constructor method
        """
        self.features = features
        self.window = window
        self.stepSize = stepSize
        # corpus for training
        self.corpus = corpus
        # unique words in corpus
        self.corpusWords = [y for x in self.corpus for y in x]
        self.corpusWords = list(set(self.corpusWords))
        self.corpusWords.sort()
        np.random.seed(42)
        # embeddings for when a word is context
        self.allContext = np.random.normal(0, .001, (len(self.corpusWords),features))
        # embeddings for when a word is center
        self.allCenter = np.random.normal(0, .001, (len(self.corpusWords),features))
        # dictionary for index of each word
        self.word2ind = self.initWord2Ind()
        # window length
        # self.concurrenceMatrix = self.computeConcurenceMatrix()

    def initWord2Ind(self):
        """
        Creating dictionary that turns a string for a word into its embedding index
        """
        word2ind = {}
        for i in range(len(self.corpusWords)):
            word2ind[self.corpusWords[i]] = i
        return word2ind
    
    def probability(self, contextIndex, centerIndex):
        """
        For a single center and context word pair:
            1. Takes the dot product of their respective embeddings
            2. Applies softmax
        """
        # embedding for context word
        context = self.allContext[contextIndex]
        # embedding for center word
        center = self.allCenter[centerIndex]
        # calculating dot product and applying softmax to normalize to probability distribution
        numerator = np.exp(np.dot(context, center))
        denominator = np.sum(np.exp(np.dot(self.allContext, center)))
        probability = numerator / denominator
        return probability

    # 'expected' output
    def contextExpected(self, centerIndex):
        """
        For a given center word:
            1. Calculates the probability of this center word and every single
            word in the corpus as context
            2. Multiplies the embedding for each context word by its respective probability
        TODO: vectorize
        """
        expected = np.empty((self.features))
        for contextWord in self.corpusWords:
            contextIndex = self.word2ind[contextWord]
            prob = self.probability(contextIndex, centerIndex)
            contextEmbedding = self.allContext[contextIndex]*prob
            expected = np.vstack((expected, contextEmbedding))
        return np.sum(expected, axis=0)
    
    # 'expected' output
    def centerExpected(self, contextIndex, centerIndex):
        prob = self.probability(contextIndex, centerIndex)
        centerEmbedding = self.allCenter[centerIndex]
        expected = centerEmbedding*prob
        return expected
    
    def contextSGD(self):
        # U = previousU - stepSize * (gradient of loss(previousU, previousV))
        pass
        
    def contextGradient(self, contextIndex, centerIndex):
        observed = self.allContext[contextIndex]
        expected = self.contextExpected(centerIndex)
        gradient = observed - expected
        return -gradient
    
    def centerGradient(self, contextIndex, centerIndex):
        observed = self.allContext[centerIndex]
        expected = self.centerExpected(contextIndex, centerIndex)
        gradient = observed - expected
        return -gradient

    # updating context word vector
    def updateContext(self, contextIndex, centerIndex):
        gradient = self.contextGradient(contextIndex, centerIndex)
        context = self.allContext[contextIndex]
        self.allContext[contextIndex] = context - self.stepSize * gradient
    
    def updateCenter(self, contextIndex, centerIndex):
        gradient = self.centerGradient(contextIndex, centerIndex)
        center = self.allCenter[contextIndex]
        self.allCenter[contextIndex] = center - self.stepSize * gradient

    def trainModel(self, epochs=1):
        for i in range(epochs):
            for work in self.corpus:
                for wordIndex, word in enumerate(work):
                    centerIndex = self.word2ind[word]
                    for windowIndex in range(-self.window, self.window+1):
                        instanceIndex = wordIndex + windowIndex
                        if (instanceIndex >= 0) and (instanceIndex < len(work)):
                            contextWord = work[instanceIndex]
                            contextIndex = self.word2ind[contextWord]
                            self.updateContext(contextIndex, centerIndex)
                            self.updateCenter(contextIndex, centerIndex)

    def predictWord(self, word):
        wordIndex = self.word2ind[word]
        centerEmbedding = self.allCenter[wordIndex]
        outputDistribution = np.dot(centerEmbedding, self.allContext.T)
        outputDistribution[wordIndex] = -10
        mostLikelyWordIndex = np.argmax(outputDistribution)
        mostLikelyWord = self.corpusWords[mostLikelyWordIndex]
        return mostLikelyWord
    
    # calculating the error of a positive sample
    # def positiveSampleError(self, contextIndex, centerIndex):
    #     probability = self.probability(contextIndex, centerIndex)
    #     error = (1 - error) * self.stepSize
    #     return error


    # def oldGradient(self, work, wordIndex):
    #     gradient = np.empty((1,self.features))
    #     centerIndex = self.word2ind[work[wordIndex]]
    #     expected = self.expected(centerIndex)
    #     # iterating through window range
    #     for windowIndex in range(-self.window, self.window+1):
    #         instanceIndex = wordIndex + windowIndex
    #         if (instanceIndex >= 0) and (instanceIndex < len(work)):
    #             observed = self.allContext[self.word2ind[work[instanceIndex]]]
    #             gradient = np.vstack((gradient, observed))
    #     return -(np.sum(gradient, axis=0) - expected)

    # def computeConcurenceMatrix(self):
    #     # initiating matrix of 0s
    #     concurenceMatrix = np.zeros((len(self.corpusWords), len(self.corpusWords)))
    #     # word counts
    #     # iterating through each unique word
    #     for word in self.corpusWords:
    #         # choosing each work in the corpus
    #         for work in self.corpus:
    #             # choosing each word in the work
    #             for corpusIndex, matchingWord in enumerate(work):
    #                 # determining if a given word in the work is what we're looking to provide a value to in the matrix
    #                 if word == matchingWord:
    #                     # checking for instances of other words within the chosen window size
    #                     for windowValue in range(-self.window,self.window+1):
    #                         instanceIndex = corpusIndex + windowValue
    #                         # ensuring the window is within bounds
    #                         if (instanceIndex >= 0) and (instanceIndex < len(work)):
    #                             instanceWord = work[instanceIndex]
    #                             # adding a value to the 'counter' in the matrix
    #                             if word != instanceWord:
    #                                 x = self.word2ind[instanceWord]
    #                                 y = self.word2ind[word]
    #                                 concurenceMatrix[x,y]+=1
    #     return concurenceMatrix


if __name__ == "__main__":
    print("does this work?")