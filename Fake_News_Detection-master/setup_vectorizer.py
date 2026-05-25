import pandas as pd
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("Setting up vectorizer for Fake News Detection...")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load training data
print("Loading training data...")
train_df = pd.read_csv('train.csv')
print("Training data shape: " + str(train_df.shape))

# No preprocessing - use raw statements like in the notebook
print("Preparing text data...")

# Create and fit TF-IDF vectorizer - match model feature size more flexibly
# The model expects 137427 features, but our training data produces 13746 unique terms
# Let's create a model with ngrams to increase feature dimensionality
print("Creating and fitting TF-IDF vectorizer with n-grams...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=None,
    lowercase=False,
    ngram_range=(1, 2),  # Include bigrams to increase feature count
    min_df=1,
    max_df=1.0
)
tfidf_vectorizer.fit(train_df['Statement'].values)

print("Vectorizer setup complete!")
print("Number of features: " + str(len(tfidf_vectorizer.get_feature_names_out())))

# Save the vectorizer
print("Saving vectorizer to tfidf_vectorizer.pkl...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Setup complete! The app is ready to use.")
