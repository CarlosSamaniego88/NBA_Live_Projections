from flask import Flask, render_template
import glob
# from get_schedule import *
# from get_team_info import *
# from get_probability import *
# from get_home_advantage import *
# from get_current_date import *
# from main import *             #fake main
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import nba_scraper.nba_scraper as ns
import itertools
import time
import numpy as np
import seaborn as sns
#import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/')
def display_predictions():
    # updated_dict = {}
    # for key in projec_d.keys():
    #     updated_dict[key.decode('utf-8')] = projec_d[key.decode('utf-8')].key.decode('utf-8')
    teams = {'static/Miami Heat.png': '53.5%', 'static/Milwaukee Bucks.png':'25.5%'}

    return render_template('home.html', teams=teams)
    # return 'hello'

if __name__=="__main__":
    app.run(debug=True)