#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load BART summarizer/reorderer
# FIGURE OUT HOW TO USE SPACEY TOKENIZATION WITH THIS?
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# In[9]:


# Set reordering function using Bart
import nltk
from nltk.tokenize import TreebankWordTokenizer

# Hyperparamers: MIN_TOKEN_MULTIPLIER, MAX_TOKEN_MULTIPLIER,
# Log base in nDCG (or discounting function as a whole)

# Trying shorter ordering
#MIN_TOKEN_MULTIPLIER = 0.8
#MAX_TOKEN_MULTIPLIER = 1

# Gives better ordering? No, does not seem to give better ordering
MIN_TOKEN_MULTIPLIER = 0.9
MAX_TOKEN_MULTIPLIER = 1.0

# Gets all sentences in output
#MIN_TOKEN_MULTIPLIER = 1.1
#MAX_TOKEN_MULTIPLIER = 1.3

# Get number of tokens using nltk
def getNumTokens(inputSentences):
    tokenizer = TreebankWordTokenizer()
    count = 0
    for sentence in inputSentences:
        count += len(tokenizer.tokenize(sentence))
    return count

# Takes in list of sentences and outputs reordered doc
def reorderBart(inputSentences):
    minLength = int(getNumTokens(inputSentences) * MIN_TOKEN_MULTIPLIER)
    maxLength = int(getNumTokens(inputSentences) * MAX_TOKEN_MULTIPLIER)
    return summarizer(" ".join(inputSentences), max_length=maxLength, min_length=minLength, do_sample=False)[0]["summary_text"]


# In[3]:


# Make sentence tokenizer with spacy
from functools import partial

import spacy

from spacy.language import Language

spacy.prefer_gpu() # depending on whether you install CPU or GPU version

def spacy_sentence_tokenizer(model: Language, text: str) -> list[str]:
    doc = model(text)
    return [sent.text.strip() for sent in doc.sents]

nlp = spacy.load('en_core_web_trf') # you need to download the gpu version of this model
spacy_tokenizer = partial(spacy_sentence_tokenizer, nlp)
#text = "I am a Naman. I study at Auburn"
#sentences = spacy_tokenizer(text) 


# In[17]:


# Sem_nDCG Metric
import copy
import math

# Return list of sentences from string document
def getSentences(doc):
    return spacy_tokenizer(doc)

# Add all possible adjacent sentence pairs to the end of the array
def addSentencePairs(sentences):
    for i in range(len(sentences) - 1):
        sentences.append(sentences[i] + " " + sentences[i + 1])

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode every sentence in list
def getEncodings(sentences):
    return [model.encode(sentence) for sentence in sentences]

# Return list of lists of cosine similiarities where the similiarity between sentence i and j are at list[i][j]
def getSimiliarities(correctSentenceEncodings, generatedSentenceEncodings):
    similiarities = []
    
    for i in range(len(correctSentenceEncodings)):
        similiarities.append([])
        for j in range(0, len(generatedSentenceEncodings)):
            similiarities[i].append(util.cos_sim(correctSentenceEncodings[i], generatedSentenceEncodings[j]))
            
    return similiarities

# Get similiarity beteen sentences at indexes i and j in given similiarity data structure
#def getSimScore(similiarities, i, j):
#    if i == j:
#        return None
#    elif i < j:
#        return similiarities[i][j-i]
#    return similiarities[j][i-j]

def getNumNonZeroes(twoDimArray):
    count = 0
    for x in range(len(twoDimArray)):
        for y in range(len(twoDimArray[x])):
            if twoDimArray[x][y] != 0:
                count += 1
    return count

def getIfZero(twoDimArray):
    for x in range(len(twoDimArray)):
        for y in range(len(twoDimArray[x])):
            if twoDimArray[x][y] != 0:
                return False
    return True    
        
# Get list of pairs of indexes, each pair is the most similiar pair found 
# up to that point wihtout repeating sentences
# TO DO, change storage of encodings so that they are in order?
def getBestPairings(similiarities):
    pairs = []
    sims = copy.copy(similiarities)
    
    #while getNumNonZeroes(sims) > 0:
    #while not getIfZero(sims):
    while True:
        maxScore = -1
        bestPairIndexes = [0, 0]
        
        for i in range(len(sims)):
            for j in range(len(sims[i])):
                if sims[i][j] > maxScore:
                    maxScore = sims[i][j]
                    bestPairIndexes = [i, j]
                    
        if maxScore == -1:
            return pairs
        
        sims[bestPairIndexes[0]] = []
        for k in range(len(sims)):
            if sims[k] != []:
                sims[k][bestPairIndexes[1]] = -1
                            
        pairs.append(bestPairIndexes)
        
    return pairs

# Reimplement functions to allow for matching adjacent sentences as a unit

# Return list of sentences with adjacent ones combined
def getSentencesWithCombinations(doc):
    sentences = getSentences(doc)
    newSentences = []
    
    for i in range(len(sentences) - 1):
        newSentences.append(sentences[i])
        newSentences.append(sentences[i] + " " + sentences[i + 1])
        
    newSentences.append(sentences[len(sentences) - 1])
    
    return newSentences

# Get list of pairs of indexes, each pair is the most similiar pair found 
# up to that point without repeating sentences
def getBestPairingsWithCombinations(similiarities):
    pairs = []
    sims = copy.copy(similiarities)
    
    #combinedCorrect = []
    #combinedReordered = []
    
    while getNumNonZeroes(sims) > 0:
        maxScore = 0
        bestPairIndexes = [0, 0]
        
        for i in range(len(sims)):
            for j in range(len(sims[i])):
                if sims[i][j] > maxScore:
                    maxScore = sims[i][j]
                    bestPairIndexes = [i, j]
                    
        # Remove two adjacent indexes too bc if we add a combined sentence then
        # its two sentences are used, and vice versa
        #combinedCorrect.append(bestPairIndexes[0])
        if bestPairIndexes[0] > 0:
            sims[bestPairIndexes[0] - 1] = []
        if bestPairIndexes[0] < len(sims) - 1:
            sims[bestPairIndexes[0] + 1] = []
        #if bestPairIndexes[1] % 2 == 1:
            #combinedReordered.append(bestPairIndexes[1])
            
        sims[bestPairIndexes[0]] = []
        for k in range(len(sims)):
            if sims[k] != []:
                sims[k][bestPairIndexes[1]] = 0
                if bestPairIndexes[1] > 0:
                    sims[k][bestPairIndexes[1] - 1] = 0
                if bestPairIndexes[1] < len(sims[k]) - 1:
                    sims[k][bestPairIndexes[1] + 1] = 0
                            
        pairs.append(bestPairIndexes)
        
    # Make corrections to pair indexes because of combined sentences
    correctIndexes = []
    reorderedIndexes = []
    
    for pair in pairs:
        correctIndexes.append(pair[0])
        reorderedIndexes.append(pair[1])
        
    sortedCorrectIndexes = sorted(correctIndexes)
    sortedReorderedIndexes = sorted(reorderedIndexes)

    numSkipped = 0
    for i in range(len(sortedCorrectIndexes)):
        pairs[correctIndexes.index(sortedCorrectIndexes[i])][0] = i + numSkipped
        
        if i != len(sortedCorrectIndexes) - 1:
            if sortedCorrectIndexes[i] % 2 == 0:
                if sortedCorrectIndexes[i + 1] - sortedCorrectIndexes[i] > 3:
                    numSkipped += 1
            else:
                if sortedCorrectIndexes[i + 1] - sortedCorrectIndexes[i] > 4:
                    numSkipped += 1
                    
    numSkipped = 0
    for i in range(len(sortedReorderedIndexes)):
        pairs[reorderedIndexes.index(sortedReorderedIndexes[i])][1] = i + numSkipped
        
        if i != len(sortedReorderedIndexes) - 1:
            if sortedReorderedIndexes[i] % 2 == 0:
                if sortedReorderedIndexes[i + 1] - sortedReorderedIndexes[i] > 3:
                    numSkipped += 1
            else:
                if sortedReorderedIndexes[i + 1] - sortedReorderedIndexes[i] > 4:
                    numSkipped += 1
        
    return pairs

# Output at 2d array where each one has the correct sentence first,
# If correct sentence is missing, put (numberOfCorrectSentences - 1) for it
def getOrderedPairs(pairs, numberOfCorrectSentences):
    orderedPairs = []
    
    for i in range(numberOfCorrectSentences):
        reorderedIndex = numberOfCorrectSentences - 1
        found = False
        
        for j in range(len(pairs)):
            if pairs[j][0] == i:
                found = True
                reorderedIndex = pairs[j][1]
                break
               
        #if not found:
            #print("Missed sentence " + str(i))
        orderedPairs.append([i, reorderedIndex])
        
    return orderedPairs

# Get nDCG score for pairs
# Uses log base 2
# ^ Tinker with log to correctly balance importance of first sentence
def nDCG(orderedPairs):
    highestIndex = len(orderedPairs) - 1
    
    correctGains = [highestIndex - pair[0] for pair in orderedPairs]
    reorderedGains = [highestIndex - pair[1] for pair in orderedPairs]
    
    numer = 0
    denom = 0
    
    for i in range(len(orderedPairs)):
        numer += reorderedGains[i] / math.log(2 + i, 2)
        denom += correctGains[i] / math.log(2 + i, 2)
        
    return numer / denom


# In[5]:


# Test sentence pairing with combinations using Bart


# In[22]:


# Get cnn_dailymail dataset

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", '3.0.0')


# In[23]:


# Test non-sentence pairing metric on cnn_dailymail dataset

import random
import copy

random.seed(0)

numArticles = 1500
numDone = 0
numSeen = 0

trainDataset = dataset["test"]["article"][0:numArticles]
del dataset

file = open("bartResults.txt", "w")
file.write("")
file.close()

for article in trainDataset:
    
    numSeen += 1
    print(numSeen)

    sentences = getSentences(article)

    correctDoc = article

    copyOfSentences = copy.copy(sentences)
    random.shuffle(copyOfSentences)
    shuffledSentences = copyOfSentences
    
    maxLength = int(getNumTokens(shuffledSentences) * MAX_TOKEN_MULTIPLIER)
    if maxLength >= 800:
        print("Doc too long")
        continue

    # Change reorder function to whatever method using
    reorderedDoc = reorderBart(shuffledSentences)
    print("Done ordering")
        
    # Looking at reordered result
    #print(correctDoc + "\n")
    #print(" ".join(shuffledSentences) + "\n")
    #print(reorderedDoc + "\n")

    correctSentences = getSentences(correctDoc)
    reorderedSentences = getSentences(reorderedDoc)

    correctEncodings = getEncodings(correctSentences)
    reorderedEncodings = getEncodings(reorderedSentences)

    simScores = getSimiliarities(correctEncodings, reorderedEncodings)
    
    bestPairs = getBestPairings(simScores)
    
    orderedPairs = getOrderedPairs(bestPairs, len(correctSentences))

    # Metric output
    numDone += 1
    print("Result for " + str(numDone))
    result = nDCG(orderedPairs)
    print(result)
    file = open("bartResults.txt", "a")
    file.write(str(result) + "\n")
    file.close()


# In[ ]:




