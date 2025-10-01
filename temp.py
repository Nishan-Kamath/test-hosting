import requests

url = "https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv"
r = requests.get(url)
with open("voice.csv", "wb") as f:
    f.write(r.content)

print("voice.csv")
