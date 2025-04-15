import requests

BOT_TOKEN = '8167802418:AAGlSOFaSYtYueGV0RdAu9AXwNqZDr7XFQQ'
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

response = requests.get(url)
print(response.json())
