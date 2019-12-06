import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import xlsxwriter
import pandas as pd
import numpy as np
import seaborn as sns

# HOME ADVANTAGE DONE BY ANISH, ANDREW

def get_home_advantage():
    pace = requests.get('https://www.teamrankings.com/nba/stat/average-scoring-margin', "html.parser")
    soup = BeautifulSoup(pace.content, "lxml")
    rows = soup.find_all('tr')
    str_cells = str(rows)

    list_rows = []
    for row in rows:
        cells = row.find_all('td')
        str_cells = str(cells)
        clean = re.compile('<.*?>')
        clean2 = (re.sub(clean, '',str_cells))
        list_rows.append(clean2)

    df = pd.DataFrame(list_rows)

    df1 = df[0].str.split(', ', expand=True)

    df1[0] = df1[0].str.strip('[')
    df1[0] = df1[0].str.strip(']')
    df1[5] = df1[5].str.strip('+')

    df1 = df1.drop(df1.index[0])
    home_margins_list = list(df1[5])

    sum = 0
    for margin in home_margins_list:
        sum += float(margin)

    home_average_margin = sum / 30

    return home_average_margin