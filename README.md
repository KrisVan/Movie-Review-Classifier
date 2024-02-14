# Movie Review Classifier
Rank and Accuracy: 112, 0.78

CS484 data mining project that performs sentiment analysis. Implementation of k nearest neighbors algorithm to classify sentiment of product reviews.
### Approach Summary
The goal of this program is to classify the sentiment of movie reviews given a training set
with +1 or -1 and a test set. The K-nearest neighbors algorithm is used to determine whether a
review in the test set is positive or negative, however, there are more steps needed to prepare the
data for this algorithm. The general steps of the program are described as follows.
1. Reading the data into a usable format.
2. Text processing.
3. Vectorizing and TFIDF.
4. Fitting the KNN.
5. Predicting the test data using KNN.
6. Result file output.

### Reading the Data into a Usable Format
Two different data sets are input into the program to get the final product, a text file of
positive and negative ones. These data sets are in the form of text files with reviews separated by
a new line. The pandas package’s read_table function was used to take each review separated by
a ‘\n’, and make it part of an indexed dataframe. The test data’s data frame consisted of the text
of each review for each index. For panda dataframes, it is possible to add multiple elements, or
columns, per index. For the training set, the sentiment at the beginning of each review was read
and put into a column to describe each review’s polarity.
### Text Processing
The purpose of processing the text is to edit the text in a way that is efficient for
tokenization and vectorization. To do this, removing words and text that are not related to
sentiment is necessary. Digits, tabs, HTML formatters, punctuation, and extra spaces are
removed from the text using re.sub. The text is also made into lowercase. This text processing
was applied to both dataframes. Another way to process the text was to stem the words into their
base form. This would reduce the amount of words that would need to be compared in the
TFIDF. It would also improve accuracy because words with the same meaning but are in
different forms would be classified as the same token. The Snowball stemmer from the nltk
package was used to do this. First, the words were tokenized into individual elements using
nltk’s word_tokenize, then the Snowball stemmer was applied to each token.
### Vectorizing and TFIDF
After the text of each review has been processed, sklearn’s TfidfVectorizer is used to get
a list of vectors for each term’s frequency. A parameter of TfidfVectorizer, max_df, will ignore a
percentage of words if they appear too often. Words that appear very frequently are likely to not
have a correlation to sentiment. For example, “the” will appear often but is just a common word
with a neutral meaning. This parameter has been adjusted to find the best way to improve
accuracy. This is explained later under the Optimizing the Model section. This is applied to the
review text of both dataframes, fit_transform being used for the training data, and transform for
the test data. Another parameter, analyzer, can be set to a function that will preprocess and
tokenize text. This was set to a function that tokenized and stemmed the text.
### Predicting the Test Data using KNN
A KNN class was made in order to store and use the data to predict the sentiment of the
test set. First, a k is selected to determine how many neighbors will be considered in choosing a
review’s sentiment. Second, the knn is fitted with the training TFIDF vectors of each review and
their sentiment. Todense is applied to the vectors to make the data not sparse. Finally, predict is
used with the test TFIDF vectors in a non-sparse format as the argument. In predict, a list of
distances between the training tfidf vectors and the test tfidf vectors is made. The closer the
distance, the more similar the reviews are. This is found using the euclidian_distance function
from sklearn. Now, for each distance, the k closest neighbors are found from the training set and
the most occurring sentiment is used to determine the test review’s polarity. The predict function
of the knn class outputs a numpy array of positive and negative ones.
### Result File Output
With the sentiment determined, the numpy array results can be converted into a txt file.
numpy.savetxt is used to do this. The parameter fmt is set to ‘%+d’, which adds a plus sign of the
positive ones. Parameter newLine is also set to ‘\n’ to put each sentiment on a newline. The
results are put into a file called output.txt.
### Optimizing the Model
The k value, the number of neighbors considered in classification, was considered in
optimizing the model. Through testing, it was determined that 75 was the optimal k value.
Increasing it any more had little effect on the accuracy of the code, while increasing run time
substantially. From the different ks tested, it can be seen that the number of ks and accuracy
follow a log path. Increasing the number of ks affects accuracy greatly when the ks are lower, but
affect it less as the ks increase.

The parameter max_df for the TFIDF vectorizer from sklearn can have values from 0 to 1. 
Max_df determines if a word should be ignored if it appears too often. For example, setting it
to 0.6 would mean if a word was found in 60% of reviews, it would be ignored. From testing
different values of this parameter, it was found that setting it to 0.6 yielded the greatest accuracy
and rank at 78% accuracy.

The porter stemmer, a more aggressive stemming algorithm than Snowball, was
considered for text processing. A test using porter stemming resulted in the same accuracy but
longer run times for the same number of k. Because of this, Snowball was used for the rest of the
testing.
