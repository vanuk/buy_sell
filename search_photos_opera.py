from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os

def search_photos_opera(url, num=50):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.binary_location = 'C:/Users/Vanyk/AppData/Local/Programs/Opera/opera.exe'  # Вкажіть правильний шлях до Opera

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    
    image_urls = []
    driver.get(url)
    time.sleep(2)
    
    while len(image_urls) < num:
        images = driver.find_elements(By.CSS_SELECTOR, "img")
        for img in images:
            src = img.get_attribute("src")
            if src and src.startswith("http") and src not in image_urls:
                image_urls.append(src)
                if len(image_urls) >= num:
                    break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    
    driver.quit()
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
    url = "https://www.google.com/search?sca_esv=fde65a004f3df79d&q=%D0%BF%D0%BE%D0%B4%D0%B2%D1%96%D0%B9%D0%BD%D0%B0+%D0%B2%D0%B5%D1%80%D1%88%D0%B8%D0%BD%D0%B0+%D1%82%D1%80%D0%B5%D0%B9%D0%B4%D0%B8%D0%BD%D0%B3&udm=2&fbs=ABzOT_DMf7N-F8a8cOkkEzmLz2vBEb33uLPKYlScLe9tioUA5igL4LwI_0XPIwc-3PIt116QFyuLgVfpE3mc-Yn9PFTXaPrtPtvMRQkTJ0Speb-U3qKNG-2GKBnSo7h-5579EAY-ny2UBOdYpdL5YO-WfxohqTOAi3rKECU8-iiZoxsGS0Tb6fU7yhTy1CSZD11On-BI1RTQPDdn-q3RF_52yoeMzpVG_CsCjcD1TO04ir6IbSTaqkZBRLZS7muozfcrwbPpIuzj&sa=X&ved=2ahUKEwjzxYaNh92LAxUQJRAIHai2AX0QtKgLegQIExAB&biw=1920&bih=953&dpr=1"
    photos = search_photos_opera(url)
    os.makedirs('downloaded_images', exist_ok=True)
    for i, photo in enumerate(photos):
        download_image(photo, f'downloaded_images/image_{i}.jpg')
        print(f'Downloaded {photo}')
