import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


stats_page = requests.get('https://www.basketball-reference.com/leagues/NBA_2020_games-november.html')

content = stats_page.content

soup = BeautifulSoup(content, 'html.parser')
table = soup.findAll(name = 'table', attrs= {'id': 'schedule'})

html_str = str(table)
df = pd.read_html(html_str)[0]

print(df.head(30))