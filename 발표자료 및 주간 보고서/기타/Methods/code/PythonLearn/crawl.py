import pandas as pd
from lxml import html
import requests
import re

print('Crawling Test')
def crawl(x):
    x=x.replace(' ','+')
    #page = requests.get('https://www.google.com/search?q='+u'\u97F3\u6A02\u6B4C\u8A5E'+'+'+x)
    page = requests.get('https://tw.search.yahoo.com/search?p='+u'\u97F3\u6A02'+'+"'+x+'"')
    tree = html.fromstring(page.content)
    #return int(tree.xpath('//div[@id="resultStats"]/text()')[0].split()[1].replace(',',''))
    #return int(tree.xpath('//div[@class="compPagination"]//span/text()')[0].split()[0].replace(',',''))
    return int(tree.xpath('//div[@class="compPagination"]//span/text()')[0].split()[0].replace(',',''))

print('Loading data...')
data_path = 'D:/musicData/'

songs = pd.read_csv(data_path + 'song_extra_info.csv',nrows=5)
songs['count'] = songs['name'].apply(crawl)
#songs=songs.drop(['name'],axis=1)
print(songs)