

def get_todays_date():
    from datetime import date
    # Getting Calendar Date
    today = date.today()
    date = today.strftime("%b %d, %Y")

    # Getting weekday
    weekdays = ['Mon, ', 'Tue, ', 'Wed, ', 'Thu, ', 'Fri, ', 'Sat, ', 'Sun, ']
    weekday = weekdays[today.weekday()]

    #Combining Weekday and Calendar Date
    todays_date = weekday + date
    #print(todays_date)


    return todays_date


