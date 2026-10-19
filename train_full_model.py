import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
import re

def clean_text(text):
    text = text.lower()                       # lowercase
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)          # remove mentions
    text = text.replace('\n',' ')             # remove newlines
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    # Keep punctuation, hashtags, emojis for sentiment
    return text

# Load full Sentiment140 CSV/ZIP
df = pd.read_csv('training.1600000.processed.noemoticon.csv.zip',
                 encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']

# Keep only positive (4) and negative (0)
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})
df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['polarity'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train models
bnb = BernoulliNB(); bnb.fit(X_train_tfidf, y_train)
svm = LinearSVC(max_iter=1000); svm.fit(X_train_tfidf, y_train)
logreg = LogisticRegression(max_iter=100); logreg.fit(X_train_tfidf, y_train)

# Save vectorizer and models
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(bnb, open('bnb_model.pkl', 'wb'))
pickle.dump(svm, open('svm_model.pkl', 'wb'))
pickle.dump(logreg, open('logreg_model.pkl', 'wb'))

print("Models trained and saved!")
