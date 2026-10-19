from flask import Flask, render_template, request
import pickle
import re
import emoji

app = Flask(__name__)

def clean_text(text):
    text = text.lower()                       # lowercase
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)          # remove mentions
    text = text.replace('\n',' ')             # remove newlines
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    # Keep punctuation, hashtags, emojis for sentiment
    return text

def count_emojis(text):
    # Works with latest emoji library versions
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Load models and vectorizer
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
bnb = pickle.load(open('bnb_model.pkl','rb'))
svm = pickle.load(open('svm_model.pkl','rb'))
logreg = pickle.load(open('logreg_model.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def home():
    predictions = {}
    final_prediction = None
    word_count = hashtag_count = emoji_count = 0

    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        vec = vectorizer.transform([cleaned_text])

        # Individual model predictions
        predictions['BernoulliNB'] = "Positive" if bnb.predict(vec)[0]==1 else "Negative"
        predictions['SVM'] = "Positive" if svm.predict(vec)[0]==1 else "Negative"
        predictions['LogisticRegression'] = "Positive" if logreg.predict(vec)[0]==1 else "Negative"

        # Majority vote for final prediction
        preds_list = list(predictions.values())
        final_prediction = max(set(preds_list), key=preds_list.count)

        # Word, hashtag, emoji counts
        word_count = len(cleaned_text.split())
        hashtag_count = cleaned_text.count('#')
        emoji_count = count_emojis(cleaned_text)

    return render_template(
        'index.html',
        predictions=predictions,
        final_prediction=final_prediction,
        word_count=word_count,
        hashtag_count=hashtag_count,
        emoji_count=emoji_count
    )

if __name__ == '__main__':
    app.run(debug=True)
