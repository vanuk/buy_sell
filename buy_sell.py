import websocket
import json
import sys
import time
from threading import Thread
from collections import defaultdict
import requests
import ssl  # Add this import

sys.stdout.reconfigure(encoding='utf-8')

THRESHOLD = 10000  # у доларах США

crypto_pairs = [
    "tstusdt"
    # "btcusdt", "ethusdt", "bnbusdt", "xrpusdt", "solusdt", "adausdt",
    # "dogeusdt", "maticusdt", "dotusdt", "ltcusdt", "shibusdt", "avaxusdt",
    # "trxusdt", "linkusdt", "xlmusdt", "uniusdt", "atomusdt", "filusdt",
    # "vetusdt", "hbarusdt", "eosusdt", "qntusdt", "manausdt", "sandusdt",
    # "apeusdt", "chzusdt", "xtzusdt", "aaveusdt", "thetausdt", "egldusdt",
    # "icpusdt", "ftmusdt", "xemusdt", "gmtusdt", "enjusdt", "zecusdt",
    # "dashusdt", "klayusdt", "crvusdt", "flowusdt", "xmrusdt", "compusdt",
    # "nearusdt", "arbusdt", "galausdt", "sushiusdt", "algousdt", "grtusdt",
    # "batusdt", "1inchusdt", "atomusdt"
]

summary = defaultdict(lambda: {"buy": 0, "sell": 0, "difference": 0})

TELEGRAM_TOKEN = '7412022099:AAF1K5yD1yRSKyge2HZEci0QVNZsDhmSpjE'
CHAT_ID = '726841637'

# Функція для відправки повідомлення в Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    return response.json()

# Функція для періодичного виведення підсумків
def print_summary():
    while True:
        time.sleep(150)  # Кожні 30 секунд
        print("\nПідсумки за останні 3 хвилини:")

        # Сортування за різницею між купленим та проданим
        sorted_summary = sorted(summary.items(), key=lambda item: item[1]["difference"], reverse=True)

        message = "Підсумки за останні 3 хвилини:\n"
        for currency, data in sorted_summary:
            difference = data["buy"] - data["sell"]
            message += f"Криптовалюта: {currency} | Куплено: {data['buy']:.2f} USD | " \
                       f"Продано: {data['sell']:.2f} USD | Різниця: {difference:.2f} USD\n"

        # Відправка підсумків в Telegram
        send_telegram_message(message)

        print("\n")
        summary.clear()

# Потік для підсумків
summary_thread = Thread(target=print_summary, daemon=True)
summary_thread.start()

def on_message(ws, message):
    data = json.loads(message)
    if 'p' in data and 'q' in data and 's' in data and 'm' in data:
        pair = data['s']  # Торгова пара
        currency = pair[:-4].upper()  # Назва криптовалюти (видалення 'usdt')
        price = float(data['p'])  # Ціна угоди
        quantity = float(data['q'])  # Кількість криптовалюти
        total = price * quantity  # Загальна сума угоди
        is_market_maker = data['m']  # Чи це маркет-мейкер

        trade_type = "Продаж" if is_market_maker else "Покупка"

        if total >= THRESHOLD:
            print(f"Велика {trade_type}: {total:.2f} USD | Криптовалюта: {currency} | "
                  f"Кількість: {quantity:.6f} | Ціна: {price:.2f}")

        # Оновлення підсумків
        if trade_type == "Покупка":
            summary[currency]["buy"] += total
        else:
            summary[currency]["sell"] += total

        # Оновлення різниці
        summary[currency]["difference"] = summary[currency]["buy"] - summary[currency]["sell"]

def on_error(ws, error):
    print(f"Помилка: {error}, Тип: {type(error)}")

def on_close(ws, close_status_code, close_msg):
    print("З'єднання закрито")

def on_open(ws):
    streams = [f"{pair}@trade" for pair in crypto_pairs]
    payload = {
        "method": "SUBSCRIBE",
        "params": streams,
        "id": 1
    }
    ws.send(json.dumps(payload))
    print("З'єднання встановлено. Відстеження великих покупок і продажів запущено...")

if __name__ == "__main__":
    socket = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # Disable SSL verification