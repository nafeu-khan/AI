import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = "Hello there! How are you doing today? NLTK is fun and powerful."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word tokenization
words = word_tokenize(text)
print("Words:", words)

# Remove stop words & punctuation
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
print("Filtered words:", filtered_words)

# Stemming
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_words]
print("Stemmed words:", stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]
print("Lemmatized words:", lemmatized)

# POS tagging
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)
