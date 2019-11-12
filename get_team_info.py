from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from selenium import webdriver

# seasons = ['2016-17', '2017-18', '2018-19', '2019-20']

#https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2019.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2018.html#all_team-stats-per_poss
#https://www.basketball-reference.com/leagues/NBA_2017.html#all_team-stats-per_poss

seasons = ['2017', '2018', '2019', '2020']

# # for season in seasons:
# stats_page = requests.get('https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss')
# content = stats_page.content

# soup = BeautifulSoup(content, 'html.parser')
# table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

# html_str = str(table)

# temp_df = pd.read_html(html_str)[0]

# schedule_df = schedule_df.append(temp_df, ignore_index = True, sort=False) 

# print(schedule_df)

#------------------------------------------------------------------------------

# d = webdriver.Chrome()
# chromedriver = '/Users/Carlos/Downloads/chromedriver'
# d = webdriver.Chrome(chromedriver)
# d.get('https://www.basketball-reference.com/leagues/NBA_2020.html#all_team-stats-per_poss')
# s = soup(d.page_source, 'html.parser').find('table', {'class':'table'})
# print(s)
# pd.read_html(str(s))
# headers, [_, *data] = [i.text for i in s.find_all('th')], [[i.text for i in b.find_all('td')] for b in s.find_all('tr')]

# final_data = [i for i in data if len(i) > 1]

# print(final_data)

#------------------------------------------------------------------------------

# NBA season we will be analyzing
# year = 2019
# URL page we will scraping (see image above)
url = "https://www.basketball-reference.com/leagues/NBA_2020.html#team-stats-per_poss::1"
# this is the HTML from the given URL
html = urlopen(url)
soup = BeautifulSoup(html, features='lxml')

# use findALL() to get the column headers
soup.findAll('tr', limit=2)
# use getText()to extract the text we need into a list
headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
# exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
headers = headers[1:]
# print(headers)

# avoid the first header row
#table = soup.findAll("table", {"class": "sortable stats_table now_sortable"})
table = soup.find(name='table', attrs={'id':'team-stats-per_poss_clone'})
theaders = soup.findAll('tr', limit=5)
#table = soup.find("table", {"class":"sortable stats_table now_sortable sliding_cols is_sorted fixed_cols"} )
print(table)
#print(theaders)
headers = [th.getText() for th in soup.findAll('tr')[0].findAll('th')]

#headers = [[table.getText() for table in table[i].findAll('th')] for i in range(len(table))]
print(headers)
#player_stats = [[table.getText() for table in rows[i].findAll('table')] for i in range(len(rows))]
#print(player_stats)

# print(rows)
print(player_stats)

stats = pd.DataFrame(player_stats, columns = headers)
print(stats.head(10))