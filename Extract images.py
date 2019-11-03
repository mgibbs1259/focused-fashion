import re
import requests
from bs4 import BeautifulSoup
import os
from selenium import webdriver

site = 'https://www.gap.com/browse/category.do?cid=5664#pageId=0&department=136&mlink=5643,17458531,DP_DD_1_W_Jeans'
directory = "gap_images/" #Relative to script location

driver = webdriver.Chrome(r'/Users/jessicafogerty/Documents/chromedriver')

driver.get(site)

soup = BeautifulSoup(driver.page_source, 'html.parser')
img_tags = soup.find_all('img')

urls = [img['src'] for img in img_tags]

for url in urls:
    #print(url)
    filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)

    with open(os.path.join(directory, filename.group(1)), 'wb') as f:
        if 'http' not in url:
            url = '{}{}'.format(site, url)
        response = requests.get(url)
        f.write(response.content)