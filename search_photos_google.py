import requests
from bs4 import BeautifulSoup
import os
import time

def search_photos_google(query, num=200):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    image_urls = []
    page = 0
    while len(image_urls) < num:
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}&start={page * 20}"
        headers = {
            "User-Agent": user_agent}
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

if __name__ == "__main__":
    query = "double top trading"
    photos = search_photos_google(query)
    os.makedirs('downloaded_images', exist_ok=True)
    for i, photo in enumerate(photos):
        download_image(photo, f'downloaded_images/image_{i}.jpg')
        print(f'Downloaded {photo}')
