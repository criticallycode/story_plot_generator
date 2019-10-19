import sqlite3
import csv

connection = sqlite3.connect("movies.db")

cursor = connection.cursor()

cursor.execute("SELECT DISTINCT plot FROM movies WHERE genre LIKE '%Horror%'")
with open('Horror_plots.csv', 'w') as p:
        plot_csv_writer = csv.writer(p)
        for row in cursor.fetchall():
            plot_csv_writer.writerow(row)

cursor.execute("SELECT DISTINCT plot FROM movies WHERE genre LIKE '%Horror%'")
with open('Horror_plots.txt', 'w') as pp:
        plot_text_writer = csv.writer(pp)
        for row in cursor.fetchall():
            plot_text_writer.writerow(row)