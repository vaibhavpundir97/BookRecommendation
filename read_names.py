from flask import Flask, render_template, request
import json
import time
import csv
import sqlite3
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

book_corr = []
book_titles = []
book_name = []
books_dataset = pd.DataFrame()
def load_data():
    global book_name
    global books_dataset
    global book_corr
    global book_titles
    con = sqlite3.connect('DataSet/books_ratings.db');
    books_dataset = pd.read_sql_query('select book_id, authors, title, average_rating, image_url from books;', con)
    book_name = books_dataset.title.tolist()
    ratings = pd.read_sql_query('select * from ratings;', con)
    #books = pd.read_csv('goodbooks-10k/books.csv')
    #ratings = pd.read_csv('goodbooks-10k/ratings.csv')
    #books_dataset = pd.DataFrame(books, columns=['book_id', 'authors', 'title', 'average_rating'])
    books_dataset = books_dataset.sort_values('book_id')
    books_data = pd.merge(books_dataset, ratings, on='book_id')
    each_book_rating = pd.pivot_table(books_data, index='user_id', values='rating', columns='title', fill_value=0)
    book_corr = np.corrcoef(each_book_rating.T)
    book_list=  list(each_book_rating)
    book_titles = []
    for i in range(len(book_list)):
        book_titles.append(book_list[i])
    print(book_titles[:10])
    return "{}"

@app.route('/books.csv/<len>')
def read_book_names(len):
    global book_name
    len = int(len)
    if len <= 9900:
        print(book_name[0])
        return json.dumps(book_name[len:len+100])
    else:
        return "{}"

@app.route('/get_recommendation', methods=["POST"])
def get_recommendation():
    global book_name
    global books_dataset
    print(request.form["book_list"])
    books_list = [book_name[int(i)-1] for i in request.form["book_list"].split(',')]
    print(books_list)
    book_similarities = np.zeros(book_corr.shape[0])

    for book in books_list:
#         print(book)
        try:
            book_index = book_titles.index(book)
        except:
            continue
#         print(book_index)
        book_similarities += book_corr[book_index]
    book_preferences = []
    for i in range(len(book_titles)):
        book_preferences.append((book_titles[i],book_similarities[i]))
    book_recommendations = sorted(book_preferences, key= lambda x: x[1], reverse=True)
    i=0
    cnt=0
    res = []
    while cnt < 9:
        book_to_read = book_recommendations[i][0]
        i += 1
        if book_to_read in books_list:
            continue
        else:
            res.append(book_to_read)
            cnt += 1
    recommendations = {}
    for i in res:
        data = tuple(books_dataset[books_dataset['title'] == i][['title', 'image_url']].values[0])
        recommendations[data[0]] = data[1]
    print(recommendations)
    return render_template('generic.html', result=recommendations)



if __name__ == '__main__':
    load_data()
    app.run(host = '0.0.0.0', debug = True)
