from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException
import pandas as pd
import re
import time

pages=[110,110,110,78,110,66]

url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100'
options=webdriver.ChromeOptions()
options.add_argument('lang=ko_KR')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')

driver=webdriver.Chrome('./chromedriver',options=options)

for i in range(0,6):
    titles=[]
    for j in range(1,pages[i]+1):
        url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i)
        driver.get(url)
        time.sleep(2)
        for k in range(1,5):
            for l in range(1,6):
                x_path='//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(k,l)
                title=driver.find_element_by_xpath(x_path).text #selenium에서 text만 출력
                print(title)