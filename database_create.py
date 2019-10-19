import sqlite3

connection = sqlite3.connect('movies_genres.db')

cursor = connection.cursor()

# only run this when creating table
sql_command = """CREATE TABLE movies (title VARCHAR, genre VARCHAR, plot VARCHAR, rated VARCHAR, year VARCHAR)"""
cursor.execute(sql_command)

connection.commit()
connection.close()