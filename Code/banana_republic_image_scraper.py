import time
import requests

from selenium import webdriver


def obtain_image_urls(driver):
    """Returns a list of image urls."""
    images = driver.find_elements_by_tag_name("img")
    image_links = []
    for image in images:
        image_url = (image.get_attribute("src"))
        if image_url.endswith(".jpg"):
            if image_url not in image_links:
                image_links.append(image_url)
                print(image_url)
    return image_links


def scrape_image_urls(driver, url_list):
    """Returns a set of all image urls scraped from a given list of urls."""
    try:
        image_urls = []
        for url in url_list:
            driver.get(url)
            time.sleep(3)
            driver.find_element_by_css_selector(".universal-modal__close-button").click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*25/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*50/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*75/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.close()
    except:
        driver.close()
    image_urls = set(image_urls)
    return image_urls


def save_images(image_urls, image_urls_type):
    """Saves the images from a given set of image urls."""
    i = 1
    for url in image_urls:
        img_data = requests.get(url).content
        f = open("{}{}.jpg".format(i, image_urls_type), "wb")
        f.write(img_data)
        f.close()
        i += 1


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-notifications")
driver = webdriver.Chrome(chrome_options=chrome_options)


url_list = [
    "https://bananarepublic.gap.com/browse/category.do?cid=69883&mlink=5001,,flyout_women_apparel_Dresses&clink=15682852"]
image_urls = scrape_image_urls(driver, url_list)
save_images(image_urls, "sweater")
