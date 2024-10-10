from time import sleep

import bleach
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from persiantools.jdatetime import JalaliDate
from selenium.webdriver.firefox.service import Service


class IRNACrawler:
    def __init__(self):
        self.base_url = 'https://www.irna.ir'
        self.service = Service(executable_path='/home/asher/Documents/Projects/news crawlers/geckodriver')
        self.options = webdriver.FirefoxOptions()
        self.options.binary_location = r'/usr/bin/firefox'
        self.driver = webdriver.Firefox(service=self.service, options=self.options)

    def crawl_pagination(self, number_of_pages, tp_map):
        month = JalaliDate.today().month
        day = JalaliDate.today().day
        year = JalaliDate.today().year
        news_detail_list = []

        driver = self.driver
        resp = driver.get(self.base_url)
        sleep(10)

        for tp in tp_map.keys():
            for i in range(1, number_of_pages + 1):
                url = f'{self.base_url}/page/archive.xhtml?mn={month}&wide=0&dy={day}&ms=0&pi={i}&yr={year}&tp={tp}'

                resp = driver.get(url)

                resp = driver.page_source
                html = BeautifulSoup(resp, 'html.parser')
                news = html.find_all('li', class_='news')
                for desc in enumerate(news):
                    print(desc)
                    news_detail_list.append({
                        'url': self.base_url + desc[1].find('h3').find('a').get('href'),
                        'topic': tp_map[tp]
                    })
        return news_detail_list

    def crawl_news_detail(self, urls):
        data_list = []
        for url in urls:
            print(url)
            self.driver.get(url.get('url'))
            resp = self.driver.page_source

            html = BeautifulSoup(resp, 'html.parser')
            text = html.find('div', {'class': 'item-text'})
            cleaned_text = bleach.clean(str(text), tags={}, strip=True)
            text = {'text': cleaned_text,
                    'topic': url.get('topic')
                    }
            sleep(5)
            data_list.append(text)
        self.create_excel(data_list)

    def create_excel(self, data):
        df = pd.DataFrame.from_dict(data)
        df.to_excel('irna.xlsx')


if __name__ == '__main__':
    crawler = IRNACrawler()
    "sport, art, economy, politics"
    tp_map = {
        15: 'sports',
        43: 'arts',
        20: 'economy',
        5: 'politics',
    }
    urls = crawler.crawl_pagination(5, tp_map)
    crawler.crawl_news_detail(urls)
    print('[done] IRNA crawler')
