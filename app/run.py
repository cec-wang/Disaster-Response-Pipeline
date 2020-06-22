# import libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('Messages', con = engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_colnames = df.columns.drop(['id', 'message', 'genre', 'original'])
    category_sum = pd.DataFrame(df[category_colnames].sum(), columns = ['count'])
    others_count = category_sum[category_sum['count']<2000].sum()['count']
    others_row = pd.DataFrame([ others_count], index = ['others'], columns = ['count'])
    category_sum = pd.concat([category_sum, others_row])
    category_sum.drop(index = category_sum[category_sum['count']<2000].index, inplace = True)
    
    top_5 = ['related', 'aid_related', 'weather_related', 'direct_report', 'request']
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
        #graph #2
         {
            'data': [
                Pie(
                    labels = list(category_sum.index),
                    values = list(category_sum['count'])
                
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
            }
        },
        # graph 3
        {
            'data': [
                Bar(
                    name = top_5[0],
                    x=genre_names,
                    y=df.groupby('genre').sum()[top_5[0]]
                ),
                Bar(
                    name = top_5[1],
                    x=genre_names,
                    y=df.groupby('genre').sum()[top_5[1]]
                ),
                Bar(
                    name = top_5[2],
                    x=genre_names,
                    y=df.groupby('genre').sum()[top_5[2]]
                ),
                Bar(
                    name = top_5[3],
                    x=genre_names,
                    y=df.groupby('genre').sum()[top_5[3]]
                ),
                Bar(
                    name = top_5[4],
                    x=genre_names,
                    y=df.groupby('genre').sum()[top_5[4]]
                ),
            ],

            'layout': {
                'title': 'Distribution of Top 5 categories under Each Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
