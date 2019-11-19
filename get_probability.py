import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_win_probability(projectedSpread):

    probabilities = pd.read_csv("spread_to_probability.csv")
    favorite_percentage = ''
    underdog_percentage = ''

    spreads = list(probabilities['Spread'])
    favorite_percentages = list(probabilities['Home'])
    underdog_percentages = list(probabilities['Away'])

    i = 0
    for i in range(len(spreads)):
        if ((projectedSpread >= spreads[i]) and (projectedSpread <= spreads[i + 1])):
            favorite_percentage = favorite_percentages[i]
            underdog_percentage = underdog_percentages[i]
        
    
    return favorite_percentage, underdog_percentage



