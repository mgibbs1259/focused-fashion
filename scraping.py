import time

from selenium import webdriver


def get_image_urls(driver):
    images = driver.find_elements_by_tag_name("img")
    image_links = []
    for image in images:
        image_url = (image.get_attribute("src"))
        if image_url.endswith(".jpg"):
            if image_url not in image_links:
                image_links.append(image_url)
                print(image_url)


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-notifications")
driver = webdriver.Chrome(chrome_options=chrome_options)
try:
    url_list = ["https://oldnavy.gap.com/browse/category.do?cid=1035712&mlink=5360,1,W_3"]
    for url in url_list:
        driver.get(url_list[0])
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*25/100);")
        time.sleep(5)
        get_image_urls(driver)
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*50/100);")
        time.sleep(5)
        get_image_urls(driver)
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*75/100);")
        time.sleep(5)
        get_image_urls(driver)
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        get_image_urls(driver)
        time.sleep(3)
except:
    driver.close()
