import numpy as np

class word2vec(object):
    # constructor
    def __init__ (self, corpus, features, window, stepSize=.025):
        """
        Constructor
        """
        self.features = features
        self.window = window
        self.stepSize = stepSize
        self.corpus = corpus
        self.corpusWords = [y for x in self.corpus for y in x]
        self.corpusWords = list(set(self.corpusWords))
        self.corpusWords.sort()
        # dictionary for index of each word
        self.word2ind = self.initWord2Ind()
        # random values to initiate embeddings
        np.random.seed(42)
        self.allContext = np.random.normal(0, .001, (len(self.corpusWords),features))
        self.allCenter = np.random.normal(0, .001, (len(self.corpusWords),features))
        
    def initWord2Ind(self):
        """
        Creating dictionary that turns a string for a word into its embedding index
        """
        word2ind = {}
        for i in range(len(self.corpusWords)):
            word2ind[self.corpusWords[i]] = i
        return word2ind
    
    def probability(self, contextEmbedding, centerEmbedding):
        """
        For a single center and context word pair:
            1. Takes the dot product of their respective embeddings
            2. Applies softmax
        """
        # calculating the numerator, which is a 1d vector the length of the number of embeddings in the function
        numerator = np.dot(contextEmbedding, centerEmbedding)
        numerator = np.exp(numerator - np.max(numerator))
        # determining denominator, which is a scalar - the same for all context words
        denominator = np.dot(self.allContext, centerEmbedding)
        denominator = np.sum(np.exp(denominator - np.max(denominator)))
        probability = numerator / denominator
        return probability

    # NOTE: this is the expected value for the context word, but is used to calculate
    # the gradient that updates the center word vector
    def contextExpected(self, centerEmbedding):
        """
        For a given center word:
            1. For each context word, calculates the (softmax) probability of that context word
            given this center word
            2. Multiplies the embedding for each context word by its respective probability
            3. Takes the sum of this matrix along the x axis, giving us a vector of length=
            number of embeddings
        The output is the 'context embedding' of the theoretical context word the model would
        predict for this center word given the current weights
        """
        prob = self.probability(self.allContext, centerEmbedding)
        expected = self.allContext * prob.reshape(len(self.corpusWords),1)
        expected = np.sum(expected, axis=0)
        return expected
    
    # NOTE: this is the expected value for the center word, but is used to calculate
    # the gradient that updates the context word vector
    def centerExpected(self, contextEmbedding, centerEmbedding):
        """
        For a given context/center word pair:
            1. Calculate the (softmax) probability of the context word given a center word
            2. Multiply this probability by the embedding for the center word
        TODO: Understand what this "represents"
        """
        prob = self.probability(contextEmbedding, centerEmbedding)
        expected = centerEmbedding * prob
        return expected
        
    def gradientWRTCenter(self, contextIndex, centerIndex):
        """
        Determine the gradient of the loss function with respect to the center word
        for a given context/center word pair:
            1. Get the observed embedding for the context word
            2. Calculate the expected context word embedding given the center word
            3. Subtract the expected from the observed
            4. Return the negative of this value
        """
        centerEmbedding = self.allCenter[centerIndex]
        observedContext = self.allContext[contextIndex]
        expectedContext = self.contextExpected(centerEmbedding)
        gradientWRTCenter = observedContext - expectedContext
        return -gradientWRTCenter
    
    def gradientWRTContext(self, contextIndex, centerIndex):
        """
        TODO: verify that the understanding of expected and observed is valid here
        Determine the gradient of the loss function with respect to the context word
        for a given context/center word pair:
            1. Get the observed embedding for the center word
            2. Calculate the expected center word embedding given the center word
            3. Subtract the expected from the observed
            4. Return the negative of this value
        """
        contextEmbedding = self.allContext[contextIndex]
        observedCenter = self.allCenter[centerIndex]
        expectedCenter = self.centerExpected(contextEmbedding, observedCenter)
        gradientWRTContext = observedCenter - expectedCenter
        return -gradientWRTContext

    def updateContext(self, contextIndex, centerIndex):
        """
        Update the context word embedding for a given context/center pair:
            1. Calculate the gradient with respect to the context word
            2. Multiply this by the step size
            3. Subtract this value from the actual context embedding
        """
        gradient = self.gradientWRTContext(contextIndex, centerIndex)
        context = self.allContext[contextIndex]
        self.allContext[contextIndex] = context - self.stepSize * gradient
    
    def updateCenter(self, contextIndex, centerIndex):
        """
        Update the center word embedding for a given context/center pair:
            1. Calculate the gradient with respect to the center word
            2. Multiply this by the step size
            3. Subtract this value from the actual center embedding
        """
        gradient = self.gradientWRTCenter(contextIndex, centerIndex)
        center = self.allCenter[contextIndex]
        self.allCenter[contextIndex] = center - self.stepSize * gradient

    def trainModel(self, epochs=1):
        """
        Update context and center word vectors while looping through the
        entire corpus
        """
        for i in range(epochs):
            for work in self.corpus:
                for wordIndex, word in enumerate(work):
                    centerIndex = self.word2ind[word]
                    for windowIndex in range(-self.window, self.window+1):
                        instanceIndex = wordIndex + windowIndex
                        if (instanceIndex >= 0) and (instanceIndex < len(work)):
                            if windowIndex != 0:
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


if __name__ == "__main__":
    print("does this work?")