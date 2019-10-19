import requests
from bs4 import BeautifulSoup
import re
import csv
import numpy as np
import time

# genres = Drama, Horror, Thriller, Mystery, Action, Sci-fi

pages = list(range(1,9))
url_base = 'https://www.imdb.com/search/title?genres=Horror&start='
url_end = '&explore=title_type,genres&ref_=adv_nxt'
print(pages)

nums = np.arange(0, 1001, 50)
print(nums)

# handle loop through by incrementing number in groups of 50

for num in nums:
    time.sleep(3)
    complete_url = url_base + str(num) + url_end
    #print(complete_url)
    print("Getting URL data")
    r = requests.get(complete_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    blocks = (soup.find_all('h3', {'class':'lister-item-header'}))
    #text = links.find_all(class_='frame')
    #print(blocks)

    for title in blocks:

        titles = []

        for y in title.find_all('a'):
            titles.append(y.text)

        print(titles)

        with open('Horror_movies.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(titles)
