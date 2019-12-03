def get_todays_date():
    from datetime import date
    # Getting Calendar Date
    today = date.today()
    date = today.strftime("%b %d, %Y")
    # print(date)
    # Getting weekday
    weekdays = ['Mon, ', 'Tue, ', 'Wed, ', 'Thu, ', 'Fri, ', 'Sat, ', 'Sun, ']
    weekday = weekdays[today.weekday()]

    #Combining Weekday and Calendar Date
    todays_date = weekday + date

    new_date = ''
    for letter in todays_date[:10]:
        if letter != '0':
            new_date += letter
    new_date += todays_date[10:]
    # print(new_date)
        
    return new_date