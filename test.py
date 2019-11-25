teams = {'static/Miami Heat.png': '53.5%', 'static/Milwaukee Bucks.png':'25.5%', 'static/Utah Jazz.png': '28.2%', 'static/Toronto Raptors.png': '80%'}

for i in range(0, len(teams), 2):
   
    print(list(teams.keys())[i])
    print(list(teams.keys())[i+1])
    print(list(teams.values())[i])
    print(list(teams.values())[i+1])
    # print(te)


            #         <td><img src="{{ team }}" alt="" style="width:100px;height:100px;">{{team[i]}}</td>
            #         <td><img src="{{ team+1 }}" alt="" style="width:100px;height:100px;"><h3>{{ team[i+1] }}</h3></td>                    
            #   {% endfor %}
            # {% endfor %}