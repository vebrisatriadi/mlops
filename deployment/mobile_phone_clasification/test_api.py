import requests

url = 'http://localhost:5050/predict'
data = {
    'features': [5000,5000,1.8,9000,50,0,5,10.1,193,3,16,226,50000,3476,9090,1234,2,3,9,10]  # Your feature values
}
response = requests.post(url, json=data)
print(response.json())