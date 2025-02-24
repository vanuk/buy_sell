import requests
from bs4 import BeautifulSoup
import os
import time
from fake_useragent import UserAgent
import tkinter as tk
from tkinter import ttk, messagebox

def search_photos_no_api(query, num=1000):
    ua = UserAgent()
    image_urls = []
    page = 0
    while len(image_urls) < num:
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}&start={page * 20}"
        headers = {
            "User-Agent": ua.random}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            images = soup.find_all("img")
            for img in images:
                src = img.get("src")
                if src and src.startswith("/"):
                    src = "https://www.google.com" + src
                if src and src not in image_urls:
                    image_urls.append(src)
                    if len(image_urls) >= num:
                        break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(120)  # Затримка перед повторною спробою
        page += 1
        time.sleep(10)  # Затримка між запитами
    return image_urls[:num]

def download_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {url}: {e}")

def start_download():
    query = "подвійна вершина"
    num = 1000
    save_path = os.getcwd()  # Поточна робоча директорія
    photos = search_photos_no_api(query, num)
    os.makedirs(save_path, exist_ok=True)
    for i, photo in enumerate(photos):
        download_image(photo, os.path.join(save_path, f'image_{i}.jpg'))
        print(f'Downloaded {photo}')
    messagebox.showinfo("Completed", "Download completed!")

# Створення графічного інтерфейсу
root = tk.Tk()
root.title("Photo Downloader")

ttk.Label(root, text="Query: подвійна вершина").grid(column=0, row=0, padx=10, pady=10)
ttk.Label(root, text="Number of Photos: 1000").grid(column=0, row=1, padx=10, pady=10)

download_button = ttk.Button(root, text="Download", command=start_download)
download_button.grid(column=0, row=2, columnspan=2, padx=10, pady=10)

root.mainloop()
