import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Day':22, 'Month':8, 'Year':2023})

print(r.json())