import selenium
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.remote.webelement import WebElement, By
from selenium.webdriver.support.ui import Select
import pandas as pd
import csv

# step1
options = webdriver.ChromeOptions()
# options.add_argument('headless')
driver = webdriver.Chrome(executable_path = "C:\chromedriver.exe", options=options)
URL = ""
list_url = []
list_review = []
list_name = []

# step2
with open(r"./urls.csv", 'r', encoding="cp949") as f:
    csvreader = csv.reader(f, delimiter=',')
    next(csvreader)

    for row in csvreader:
        url = row[1]
        name = row[0]

        driver.get(url)
        time.sleep(3)
        driver.find_element_by_xpath('//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span/span[2]/span[1]/button').click()


        # step3
        time.sleep(2)
        scroll_cnt = 300

        scrollable_div = driver.find_element_by_css_selector(
            '#pane > div > div.widget-pane-content.cYB2Ge-oHo7ed > div > div > div.section-layout.section-scrollbox.cYB2Ge-oHo7ed.cYB2Ge-ti6hGc')

        for i in range(scroll_cnt):
            driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
            time.sleep(1)

        response = BeautifulSoup(driver.page_source, 'html.parser')
        rblock = response.find_all('span', {'class': 'ODSEW-ShBeI-text'})

        for index, review in enumerate(rblock):
            review = review.get_text()
            review = review.replace('(Google 번역 제공)', '')
            review = review.replace('(원문)', '')
            review = review.replace('\n', '')
            review = review.strip()

            list_url.append(url)
            list_review.append(review)
            list_name.append(name)

            print(review)

df = pd.DataFrame(list(zip(list_name, list_review, list_url)), columns = ['name', 'review', 'url'])
df.to_csv("./data/review.csv",index=True, encoding='utf-8-sig')

f.close()
driver.close()
