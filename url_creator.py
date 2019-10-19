import csv
import re

url_base = "http://www.omdbapi.com/?apikey=[YOUR API KEY GOES HERE]&t="

with open('Horror_movies_correct.csv', mode='r') as data:
    reader = csv.reader(data)
    titles = list(reader)

print(titles)

translation_table1 = dict.fromkeys(map(ord, "]'["), None)

for title in titles:
   concat_url = url_base + str(title)
   concat_url2 = str(concat_url).translate(translation_table1)
   concat_url3 = concat_url2.replace(" ", "+")
   concat_url3 = re.sub(r'"', '', concat_url3)

   #print(concat_url3)

   urls = []
   urls.append(concat_url3)

   print(urls)

   with open ('Horror_urls.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(urls)