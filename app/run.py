import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objects import Scatter
import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


app = Flask(__name__)

def tokenize(text):
    """utilizes nltk.tokenize method word_tokenize(text) and returns the tokens as a list"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)
cv = CountVectorizer()
text_data = []
for each in df.message:
    text_data.append(each)
d = cv.fit_transform(text_data)
scatter_x = pd.Series(cv.get_feature_names())
scatter_y = pd.DataFrame(d.toarray())
#print(scatter_x)
#print("TEST: ", scatter_y.sum(axis=0))
scatter_y = scatter_y.sum(axis=0)
count = 0
Final_DF = pd.DataFrame(data = scatter_y, index = scatter_x, columns = ['count'])
count = 0
for each in scatter_y:
    Final_DF.iloc[count, 0] = each
    count += 1

df3 = Final_DF.sort_values(by=['count'], ascending=False).head(35)
x_2 = df3.index
y_2 = df3['count']

#print('Y: ', scatter_y)
#new_df = pd.concat([scatter_x, scatter_y])
#print(new_df.dropna())

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """non returning function creating the plotly visual utilized on index.html page"""
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    genre_counts = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=y_2,
                    y=x_2,
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Top Frequency Words',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'word-token'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """web page that handles user query and displays model results, returns render_template()"""
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict(tokenize([query]))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """main method"""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()