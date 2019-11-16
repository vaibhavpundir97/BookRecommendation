from flask import Flask, render_template, request
import json
import time
import csv
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

book_name = []
@app.route('/books.csv')
def read_book_names():
    global book_name
    with open("goodbooks-10k/books.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            book_name.append(dict(row)['title'])
    csvfile.close()
    print(book_name[0])
    return json.dumps(book_name)

book_corr = []
book_titles = []
@app.route('/load_data')
def load_data():
    books = pd.read_csv('goodbooks-10k/books.csv')
    ratings = pd.read_csv('goodbooks-10k/ratings.csv')
    books_dataset = pd.DataFrame(books, columns=['book_id', 'authors', 'title', 'average_rating'])
    books_dataset = books_dataset.sort_values('book_id')
    books_data = pd.merge(books_dataset, ratings, on='book_id')
    each_book_rating = pd.pivot_table(books_data, index='user_id', values='rating', columns='title', fill_value=0)
    global book_corr
    book_corr = np.corrcoef(each_book_rating.T)
    book_list=  list(each_book_rating)
    global book_titles
    book_titles = []
    for i in range(len(book_list)):
        book_titles.append(book_list[i])
    return "{}"

@app.route('/get_recommendation', methods=["POST"])
def get_recommendation():
    books_list = [book_name[int(i)-1] for i in request.json]
    book_similarities = np.zeros(book_corr.shape[0])

    for book in books_list:
#         print(book)
        book_index = book_titles.index(book)
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
    print(res)
    return json.dumps(res)

if __name__ == '__main__':
   app.run(host = 'localhost', debug = True)