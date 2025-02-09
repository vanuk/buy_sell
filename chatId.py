import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Введіть токен вашого бота
TOKEN = "7412022099:AAF1K5yD1yRSKyge2HZEci0QVNZsDhmSpjE"

def get_updates():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()

    if "result" in data and len(data["result"]) > 0:
        for update in data["result"]:
            chat_id = update["message"]["chat"]["id"]
            print(f"Ваш chat_id: {chat_id}")
    else:
        print("Не знайдено жодного повідомлення. Відправте щось боту!")

if __name__ == "__main__":
    get_updates()
