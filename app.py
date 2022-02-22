from flask import Flask, render_template, request
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
from bs4 import BeautifulSoup


def clean_tokenize_stop(text):
    
    #Removes Unnecessary characters and only collects alphabets or numbers
    cleaned_text = BeautifulSoup(text, features='lxml').get_text()
    
    characters = r"(\w+|\w!|!|\d)"
    cleaned = regexp_tokenize(cleaned_text, characters) #transform text into lower cases
        
    #Removing stopwords or common words which dont add any meaning
    stopwords_ = stopwords.words("english")
    cleaned2 = [item for item in cleaned if item not in stopwords_]
    
    cleaned3 = [PorterStemmer().stem(word) for word in cleaned2]
    cleaned4 = [WordNetLemmatizer().lemmatize(word) for word in cleaned3] # running runs
    
    cleaned5 = ' '.join([word for word in cleaned4]) 
                
    return cleaned5

model = joblib.load('model/finalized_model.pkl')
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict():
    text = request.form['review']
    text = clean_tokenize_stop(text)
    final_text = [text]
    output = model.predict(final_text)[0]
    
    if output == 0:
        out = 'This is a legit email!'
    else:
        out = 'This is definitely a spam!'
        
    return render_template('index.html', prediction_text = out)
     
  

if __name__ == '__main__':  
    app.run(debug=False)