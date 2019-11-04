import time
import os
os.system("pip3 install requests")
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
    return image_links


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-notifications")
driver = webdriver.Chrome(options=chrome_options)


try:
    url_list = ["https://bananarepublic.gap.com/browse/category.do?cid=69883&mlink=5001,,flyout_women_apparel_Dresses&clink=15682852"]
    image_urls = []
    for url in url_list:
        driver.get(url_list[0])
        time.sleep(3)
        driver.find_element_by_css_selector(".universal-modal__close-button").click()
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*25/100);")
        time.sleep(5)
        image_urls += get_image_urls(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*50/100);")
        time.sleep(5)
        image_urls += get_image_urls(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight*75/100);")
        time.sleep(5)
        image_urls += get_image_urls(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        image_urls += get_image_urls(driver)
        driver.close()
except:
    driver.close()

image_urls = (set(image_urls))
#image_urls = filter(lambda x: 'asset' in x, image_urls)


#opens url, gets image, saves image.
import requests
i=1
for url in image_urls:
    img_data = requests.get(url).content
    f = open(f"{i}dresses.jpg", "wb")
    f.write(img_data)
    f.close()
    i+=1


