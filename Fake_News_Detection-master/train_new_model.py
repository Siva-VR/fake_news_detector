import pandas as pd
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("Training a new model compatible with the current vectorizer...")

# Load training data
print("Loading training data...")
train_df = pd.read_csv('train.csv')
print("Training data shape: " + str(train_df.shape))

# Create and fit vectorizer with n-grams
print("Creating and fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=None,
    lowercase=False,
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0
)

# Vectorize training data
X_train = tfidf_vectorizer.fit_transform(train_df['Statement'].values)
print("Feature matrix shape: " + str(X_train.shape))

# Prepare labels - the labels are boolean True/False
# Convert to 1/0 for the classifier
Y_train = train_df['Label'].astype(int).values
print("Labels shape: " + str(Y_train.shape))
print("Unique label values: " + str(set(Y_train)))

# Train the classifier
print("Training PassiveAggressiveClassifier...")
classifier = PassiveAggressiveClassifier(max_iter=50, random_state=42, n_jobs=-1)
classifier.fit(X_train, Y_train)

# Check training score
train_score = classifier.score(X_train, Y_train)
print("Training accuracy: {:.4f}".format(train_score))

# Save both the vectorizer and the model
print("\nSaving vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Saving new model as model_new.pkl...")
with open('model_new.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("\nTraining complete!")
print("Models saved as:")
print("  - tfidf_vectorizer.pkl")
print("  - model_new.pkl")
print("\nTo use the new model in front.py, update it to load 'model_new.pkl' instead of 'model.pkl'")
