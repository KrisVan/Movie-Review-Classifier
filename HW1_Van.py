# -------------------------------------------------------------------------------
# Name: Kristopher Van
# Homework 1
# Movie Review Classification
# Due Date: 2/17/22
# -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.metrics.pairwise import euclidean_distances
from statistics import mode

def process(text):
	"""
		Format text so that punctuations, digits, /t, and other text
		formatters are removed. Also lowercase text.
		Returns the text.
	"""
	# Remove all digits.
	text = text.translate(text.maketrans('', '', string.digits))
	# Remove tabs and lowercase.
	text = re.sub('\t', '', text).lower()
	# Remove HTML
	text = re.compile(r'<.*?>').sub('', text)
	text = re.sub(r'[-()]', '', text)
	# Remove punctuation
	text = re.sub(r'[^\w\s]', ' ', text)
	# Remove extra spaces
	text = re.sub(' +', ' ', text).strip()
	return text


def stemmer(text):
	"""
		Stem words to their base form. Turn each word from text into a token
		and stem each token.
	"""
	# Tokenize text
	tokens = word_tokenize(text)
	# Stem
	snowStem = SnowballStemmer(language='english')
	# Create new list for stemmed words
	tokens = [snowStem.stem(token) for token in tokens]

	return tokens

class knn():
	def __init__(self, k):
		"""
			Use k nearest neighbors algorithm to classify data.
		"""
		self.k = k
		self.trainVectors = None
		self.trainClassification = None

	def fit(self, x, y):
		"""
			Set training data for KNN.
			x are the tfidf vectors and y are sentiments to compare to.
		"""
		self.trainVectors = x
		self.trainClassification = y

	def predict(self, x):
		"""
			Predict the class labels for training data given 
			x from the test data.
		"""
		# Get distances between training and test data
		#distanceList = np.linalg.norm(self.trainVectors, x) FIXME
		distanceList = euclidean_distances(x, self.trainVectors)

		# Predict neighbors
		predictionList = np.zeros(distanceList.shape[0], dtype='int8')
		for i, distances in enumerate(distanceList, start=0):
			neighborsList = []
			# Sort distances in i by their indices
			sortedDistances = np.argsort(distanceList[i])

			# Find k closest neighbors and their sentiment.
			for j in range(self.k):
				neighborsList.append(self.trainClassification[sortedDistances[j]])
			# Predict based on the k neighbors' sentiment
			predictionList[i] = mode(neighborsList)
		return predictionList

def getSentiment(rawData):
	"""
		Get sentiment from the training data and return a list of
		positive and negative 1s.
	"""
	# Get the rating from training data as a list
	ratingList = np.zeros(rawData.shape[0], dtype = 'int8')
	data = rawData.itertuples(index=True, name=None)
	for index, row in data:
		if '-1' in row[:2]:
			ratingList[index] = -1
		else:
			ratingList[index] = 1
	return ratingList

def main():
	# Parameters
	k = 75
	size = 15000 # 15000
	
	# Import the training/test data and tfIDF vectorize it. Derive labels from training data.
	print('Converting Data Sets to Dataframes')	
	trainData = pd.read_table('train_file.txt', sep="\n", header=None, names=['Reviews'], dtype=str, nrows=size)
	trainData['Sentiment'] = getSentiment(trainData)
	testData = pd.read_table('test_file.txt', sep="\n", header=None, names=['Reviews'], dtype=str, nrows=size)
	
	# Preprocessing 
	print('\nText Processing')
	trainData['Reviews'] = trainData['Reviews'].apply(lambda x: process(x))
	testData['Reviews'] = testData['Reviews'].apply(lambda x: process(x))
	
	# Vectorize and stem
	print('\nVectorizing and Stemming')
	vectorizer = TfidfVectorizer(max_df = 0.6, lowercase=False, analyzer=stemmer)
	trainVectors = vectorizer.fit_transform(trainData['Reviews'])
	testVectors = vectorizer.transform(testData['Reviews'])

	# KNN and fitting training data/sentiment
	print('\nFitting KNN')
	knnClass = knn(k=k)
	knn.fit(knnClass, trainVectors.todense(), trainData['Sentiment'])
	
	# Generate list of predictions
	print('\nPredicting Test Data')
	predictionList = knn.predict(knnClass, testVectors.todense())

	# Save predictions to output text file
	print('\nWriting Results to output.txt ')
	print('List: ', predictionList)
	file = open('output.txt','w')
	np.savetxt(fname='output.txt', X=predictionList, fmt='%+d', newline='\n')
	file.close()


if __name__ == "__main__":
	main()
