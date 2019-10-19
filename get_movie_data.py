import sqlite3
import requests
import csv
import re
from simplejson.errors import JSONDecodeError

# create the database
connection = sqlite3.connect("movies.db")

cursor = connection.cursor()

# get API data

file = open('Horror_urls_2_correct.csv')
csv_file = csv.reader(file)

for row in csv_file:
    try:
        url = re.sub(r'.*http', 'http', row[0])
        response = requests.get(url).json()
        print(response)

        title = response['Title']
        genre = response['Genre']
        year = response['Year']
        rating = response['Rated']
        plot = response['Plot']

        print("titles is:", title)
        print("genres are:", genre)

        # title, genre, plot, rated, year
        cursor.execute("INSERT INTO movies (title, genre, plot, rated, year) VALUES (?, ?, ?, ?, ?)", (title, genre, plot, rating, year))

        connection.commit()
        print("Added to database")
    except (KeyError, JSONDecodeError):
        continue


connection.close()