
import csv
dat,rdat, sdat=[],[],[]

with open('BX-Books.csv', encoding="latin-1") as o:
    rd=csv.reader(o, delimiter=';')
    names=next(rd)
    for i in rd:dat.append(i)

#print(dat[0], len(dat[0][0].split(';')))

with open('BX-Book-Ratings.csv', encoding="latin-1") as o:
    rd=csv.reader(o, delimiter=';')
    names=next(rd)
    for i in rd:rdat.append(i)

with open('BX-Users.csv', encoding="latin-1") as o:
    rd=csv.reader(o, delimiter=';')
    names=next(rd)
    for i in rd:sdat.append(i)

import sqlite3


db=sqlite3.connect("BOOKS1.db")


db.execute("""create table BOOKS('ISBN' TEXT,
'Book-Title' TEXT,
'Book-Author' TEXT,
'Year-Of-Publication' TEXT,
'Publisher' TEXT,
'Image-URL-S' TEXT,
'Image-URL-M' TEXT,
'Image-URL-L' TEXT
);""")

db.execute("""create table RATINGS('User-ID' TEXT,
'ISBN' TEXT,
'Book-Rating' INT
);""")

db.execute("""create table USERS('USER-ID' TEXT,
'Location' TEXT,
'AGE' INT
);""")

qsql="insert into BOOKS values(?"+(",?"*7)+");"
sime = 0
for i in dat:
    db.execute(qsql,i)
    sime += 1

rsql="insert into RATINGS values(?,?,?);"
for i in rdat:db.execute(rsql,i)

ssql = "insert into USERS values(?,?,?);"
for i in sdat:db.execute(ssql, i)
db.commit()
