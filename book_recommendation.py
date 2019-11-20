from flask import Flask, render_template, request
import json
from random import sample
import time
import csv
import sqlite3
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html', result = True)

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/verify_login', methods=['POST'])
def verify_login():
    email = request.form['email'];
    passwd = request.form['pass'];
    con = sqlite3.connect('DataSet/BOOKS.db')
    cursor = con.cursor()
    query = "select ID, Password from USERS_INFO where email = (?);"
    cursor.execute(query, (email,))
    print(email, passwd)
    pwd = cursor.fetchall()
    print(pwd)
    for p in pwd:
        print(p)
        if p[1] == passwd:
            return render_template('recommendation.html', result = [p[0]])
    cursor.close()
    return render_template('login.html')

@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')

@app.route('/signup_details', methods=["POST"])
def signup_details():
    name = request.form['name']
    email = request.form['email']
    username = request.form['username']
    passwd = request.form['pass']
    con = sqlite3.connect('DataSet/BOOKS.db');
    cursor = con.cursor()
    insert = "insert into 'USERS_INFO' ('Name', 'Email', 'USER-ID', 'Password') values (?, ?, ?, ?);"
    cursor.execute(insert, (name, email, username, passwd))
    con.commit()
    cursor.close()
    print(name, email, username, passwd)
    return render_template('login.html')

@app.route('/logout.html')
def logout():
    return render_template('index.html', result = True)

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
    #print(book_titles[:10])
    return "{}"

@app.route('/books.html/<len>')
def read_book_names(len):
    global book_name
    len = int(len)
    if len <= 9900:
        print(book_name[0])
        return json.dumps(book_name[len:len+100])
    else:
        return "{}"

@app.route('/recommendations.html', methods=["POST"])
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
    query = {}
    for i in books_list:
        data = tuple(books_dataset[books_dataset['title'] == i][['title', 'image_url']].values[0])
        query[data[0]] = data[1]
    print(query)
    return render_template('generic.html', result=[recommendations, query])

#New DataSet

def preload():
    global books, users, ratings
    con = sqlite3.connect('DataSet/BOOKS.db')
    books = pd.read_sql_query('select * from BOOKS;', con)
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    users = pd.read_sql_query('select * from USERS;', con)
    users.columns = ['userID', 'Location', 'Age']
    ratings = pd.read_sql_query('select * from RATINGS;', con)
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    con.close()
    cleaningbooks()
    cleaningusers()
    cleaningratings()
    filterdata()

def cleaningbooks():
    #From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
    #ISBN '0789466953'
    global books
    books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
    books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
    books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
    books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
    #ISBN '078946697X'
    books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
    books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
    books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
    books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
    #making required corrections as above, keeping other fields intact
    books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
    books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
    books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
    books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
    #Correcting the dtypes of yearOfPublication
    books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
    #However, the value 0 is invalid and as this dataset was published in 2004, I have assumed the the years after 2006 to be
    #invalid keeping some margin in case dataset was updated thereafer
    #setting invalid years as NaN
    books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN
    #replacing NaNs with mean value of yearOfPublication
    books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
    #resetting the dtype as int32
    books.yearOfPublication = books.yearOfPublication.astype(np.int32)
    #since there is nothing in common to infer publisher for NaNs, replacing these with 'other
    books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
    books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'

def cleaningusers():
    global users
    users.loc[(users.Age == 'NULL'), 'Age'] = '-1'
    users.Age = users.Age.astype(np.int32)
    #In my view values below 5 and above 90 do not make much sense for our book rating case...hence replacing these by NaNs
    users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
    #replacing NaNs with mean
    users.Age = users.Age.fillna(users.Age.mean())
    #setting the data type as int
    users.Age = users.Age.astype(np.int32)

def cleaningratings():
    global ratings, ratings_explicit
    #ratings dataset should have books only which exist in our books dataset, unless new books are added to books dataset
    ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
    #ratings dataset should have ratings from users which exist in users dataset, unless new users are added to users dataset
    ratings = ratings[ratings.userID.isin(users.userID)]
    #Hence segragating implicit and explict ratings datasets
    ratings_explicit = ratings_new[ratings_new.bookRating != 0]
    ratings_implicit = ratings_new[ratings_new.bookRating == 0]
    #Similarly segregating users who have given explicit ratings from 1-10 and those whose implicit behavior was tracked
    users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
    users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]


@app.route('/popularity.html')
def popularitybasedrecommendation():
    #At this point , a simple popularity based recommendation system can be built based on count of user ratings for different books
    ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
    top10 = ratings_count.sort_values('bookRating', ascending = False).head(10)
    ##print "Following books are recommended"
    top10.merge(books, left_index = True, right_on = 'ISBN')
    #need to implement render_template

"""@Collaborative Filtering"""

valid_user_ids = ['100459' '100906' '101209' '101606' '101851' '102359' '102647' '102702'
 '102967' '104399' '104636' '105028' '105517' '10560' '105979' '106007'
 '107784' '107951' '109574' '109901' '109955' '110483' '110912' '110934'
 '110973' '112001' '113270' '113519' '114368' '114868' '114988' '115002'
 '115003' '116599' '11676' '117384' '11993' '120565' '122429' '122793'
 '123094' '123608' '123883' '123981' '12538' '125519' '125774' '126492'
 '126736' '127200' '127359' '12824' '128835' '129074' '129716' '12982'
 '129851' '130554' '130571' '132492' '132836' '133747' '134434' '135149'
 '135265' '13552' '136010' '136139' '136348' '136382' '13850' '138578'
 '138844' '140000' '140358' '141902' '142524' '143175' '143253' '143415'
 '14422' '145449' '146113' '146348' '147847' '148199' '148258' '148744'
 '148966' '149907' '149908' '150979' '153662' '15408' '15418' '156150'
 '156269' '156300' '156467' '157247' '157273' '158226' '158295' '158433'
 '159506' '160295' '162052' '162639' '162738' '163759' '163761' '163804'
 '163973' '164096' '164323' '164533' '164828' '164905' '165308' '165319'
 '165758' '166123' '16634' '166596' '16795' '168047' '168245' '16966'
 '169682' '170513' '170634' '171118' '172030' '172742' '172888' '173291'
 '173415' '174304' '174892' '177072' '177432' '177458' '178522' '17950'
 '179718' '179978' '180378' '180651' '181176' '182085' '182086' '182993'
 '183958' '183995' '184299' '184532' '185233' '185384' '187145' '187256'
 '187517' '189139' '189334' '189835' '189973' '190708' '19085' '190925'
 '193458' '193560' '193898' '194600' '196077' '196160' '196502' '197659'
 '199416' '200226' '201290' '203240' '2033' '204864' '205735' '205943'
 '206534' '207782' '208406' '208671' '209516' '21014' '210485' '2110'
 '211426' '211919' '212965' '214786' '216012' '216444' '216683' '217106'
 '217318' '217740' '218552' '218608' '219546' '219683' '222204' '222296'
 '223087' '223501' '224349' '224525' '224646' '224764' '225087' '225199'
 '225232' '225595' '225763' '226965' '227250' '227447' '227520' '2276'
 '227705' '229011' '229329' '229551' '229741' '230522' '231210' '232131'
 '232945' '233911' '234359' '234828' '235105' '235282' '235935' '236058'
 '236283' '236340' '236757' '236948' '23768' '23872' '23902' '239584'
 '239594' '240144' '240403' '240543' '240567' '240568' '241198' '241666'
 '241980' '242006' '242083' '242409' '242465' '244627' '244685' '245410'
 '245827' '246311' '247429' '247447' '248718' '249894' '250405' '250709'
 '251394' '251843' '251844' '252695' '252820' '25409' '254206' '254465'
 '254899' '255489' '25601' '257204' '258152' '258185' '258534' '25981'
 '261105' '261829' '262998' '264031' '264082' '264321' '264525' '265115'
 '265313' '26535' '26544' '26583' '265889' '266056' '266226' '268110'
 '268300' '268932' '269566' '270713' '271448' '271705' '273113' '274061'
 '274301' '275970' '277427' '278418' '28591' '28634' '29259' '30276'
 '30511' '30711' '30735' '30810' '31315' '31556' '31826' '32773' '33145'
 '35433' '35836' '35857' '35859' '36299' '36554' '36606' '36609' '36836'
 '36907' '37644' '37712' '37950' '38023' '38273' '38281' '39281' '39467'
 '4017' '40889' '40943' '43246' '4385' '43910' '46398' '47316' '48025'
 '48494' '49144' '49889' '51883' '52199' '52350' '52584' '52614' '52917'
 '53220' '55187' '55490' '55492' '5582' '56271' '56399' '56447' '56554'
 '56959' '59172' '60244' '60337' '60707' '6242' '6251' '63714' '63956'
 '65258' '6543' '6575' '66942' '67840' '68555' '69078' '69389' '69697'
 '70415' '70594' '70666' '72352' '7286' '7346' '73681' '75591' '75819'
 '76151' '76223' '76499' '76626' '78553' '78783' '78834' '78973' '79441'
 '8067' '81492' '81560' '8245' '83287' '83637' '83671' '85526' '85656'
 '86189' '8681' '86947' '87141' '87555' '88283' '88677' '88693' '88733'
 '8890' '89602' '91113' '92652' '92810' '93047' '93363' '93629' '94242'
 '94347' '94853' '94951' '95010' '95359' '95902' '95932' '96448' '97754'
 '97874' '98391' '98758']
#setting global variables
global metric,k
k=3
metric='cosine'

def filterdata():
    #To cope up with computing power I have and to reduce the dataset size, I am considering users who have rated atleast 100 books
    #and books which have atleast 100 ratings
    global ratings_explicit, ratings_matrix, average_rating, ratings_count
    counts1 = ratings_explicit['userID'].value_counts()
    ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
    counts = ratings_explicit['bookRating'].value_counts()
    ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]
    #Generating ratings matrix from explicit ratings table
    ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
    userID = ratings_matrix.index
    ISBN = ratings_matrix.columns
    print(ratings_matrix.shape)
    ratings_matrix.head()
    #Notice that most of the values are NaN (undefined) implying absence of ratings
    #since NaNs cannot be handled by training algorithms, replacing these by 0, which indicates absence of ratings
    #setting data type
    ratings_matrix.fillna(0, inplace = True)
    ratings_matrix = ratings_matrix.astype(np.int32)
    t = ratings.groupby('ISBN', as_index=False)['bookRating'].mean()
    average_rating = dict(zip(t['ISBN'].values, t['bookRating'].values))
    ratings_count = ratings.ISBN.value_counts()
    #print(ratings_matrix.index.values)

#This function predicts rating for specified user-item combination based on user-based approach
#@app.route('userbased.html')
def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0
    prediction = int(round(mean_rating + sum([(ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:]))*similarities[i] for i in range(len(indices.flatten())) if indices.flatten()[i] != user_loc])/sum_wt))
    '''for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else:
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product'''

    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below
    if prediction <= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10

    #prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    #print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)

    return prediction

#This function finds k similar users given the user_id and ratings matrix
#These similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'auto', leaf_size=20)
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices


#This function finds k similar items given the item_id and ratings matrix

def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]
    ratings=ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'auto', leaf_size=20)
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices

#This function predicts the rating for specified user-item combination based on item-based approach
#@app.route('itembased.html')
def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    prediction = int(round(sum([ratings.iloc[user_loc,indices.flatten()[i]] * (similarities[i]) for i in range(len(indices.flatten())) if indices.flatten()[i] != item_loc])/sum_wt))
    '''for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))'''

    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    #predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10

    #print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)

    return prediction

#This function utilizes above functions to recommend items for item/user based approach and cosine/correlation.
#Recommendations are made if the predicted rating for an item is >= to 6,and the items have not been rated already
@app.route('/autorecommendation.html', methods = ["POST"])
def recommendItem():
    #print("in")
    ratings = ratings_matrix
    user_id = str(request.json[0])
    #print(type(user_id))
    approach = request.json[1]
    metric = sample(['correlation', 'cosine'], 1)[0]
    '''if (user_id not in ratings.index.values) or type(user_id) is not int:
        print("User id should be a valid integer from this list :\n\n {} ".format(re.sub('[\[\]]', '', np.array_str(ratings_matrix.index.values))))
    else:'''
    #ids = ['Item-based (correlation)','Item-based (cosine)','User-based (correlation)','User-based (cosine)']
    #select = widgets.Dropdown(options=ids, value=ids[0],description='Select approach', width='1000px')
    #def on_change(change):
    #clear_output(wait=True)
    #if change['type'] == 'change' and change['name'] == 'value':
    """if (approach == 'Item-based (correlation)') | (approach == 'User-based (correlation)') :
        metric = 'correlation'
    else:
        metric = 'cosine'"""
       # with suppress_stdout():
    #print("approach")
    if (approach == 'item'):
        #print("in app")
        prediction = [predict_itembased(user_id, str(ratings.columns[i]) ,ratings, metric) if ratings[str(ratings.columns[i])][user_id] !=0 else -1 for i in range(ratings.shape[1])]
        '''for i in range(ratings.shape[1]):
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                prediction.append(predict_itembased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:
                prediction.append(-1) #for already rated items'''
    else:
        #print("in app")
        prediction = [predict_userbased(user_id, str(ratings.columns[i]) ,ratings, metric) if ratings[str(ratings.columns[i])][user_id] !=0 else -1 for i in range(ratings.shape[1])]
        '''for i in range(ratings.shape[1]):
            #print("in for")
            #print(ratings[str(ratings.columns[i])][4385])
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                #print("in if")
                prediction.append(predict_userbased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:
                #print("in else")
                prediction.append(-1) #for already rated items'''
    #print("prediction")
    prediction = pd.Series(prediction)
    prediction = prediction.sort_values(ascending=False)
    recommended = prediction[:10]
    #print("recommend")
    #print("As per {0} approach....Following books are recommended...".format(approach))
    recommendations = []
    for i in range(len(recommended)):
        data = list(books.loc[recommended.index[i], ['ISBN', 'bookTitle', 'imageUrlL']].values)
        try:
            data.append(int(ratings_count[data[0]]))
            data.append(int(average_rating[data[0]]))
        except:
            data.append(2)
            data.append(sum(average_rating.values())/len(average_rating))
        recommendations.append(data[1:])
    '''for i in range(len(recommended)):
         print("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]]))'''
    return json.dumps(recommendations)
    #select.observe(on_change)
    ##display(select)

if __name__ == '__main__':
    load_data()
    print("loaded 1")
    preload()
    print("loaded 2")
    app.run(host = '0.0.0.0', debug = True)
