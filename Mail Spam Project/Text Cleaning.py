
############ Header files for Data Cleaning ###################
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

##########  Header file for vectorization   ##################
from sklearn.feature_extraction.text import TfidfVectorizer


def MyTokenizer(sentence):
	
	# Regular Expression tokenization
	tokenizer = RegexpTokenizer("[a-zA-Z0-9]+")
	word_list = tokenizer.tokenize(sentence.lower())

	# Stopwords Removal
	sw = set(stopwords.words("english"))
	word_list = [w for w in word_list if w not in sw]

	#Stemming
	final_list = []
	ss = SnowballStemmer("english")

	for word in word_list:
		new_word = ss.stem(word)
		if new_word not in final_list:
			final_list.append(new_word)

	return final_list




corpus = [   
    'Indian cricket team will win World Cup ,says Capt. Virat Kohli',
    'We will win next Lok Sabha Elections, says confident Indian PM ',
    'The nobel laurate won the hearts of the people',
    'The movie Raazi is an exciting Indian Spy thriller based upon a real story.'
]



tfidf_vectorizer = TfidfVectorizer(tokenizer = MyTokenizer,ngram_range=(1,2),norm='l2')
vectorized_corpus = tfidf_vectorizer.fit_transform(corpus).toarray()
print(vectorized_corpus)
print('\n\n\n')
print(tfidf_vectorizer.vocabulary_)