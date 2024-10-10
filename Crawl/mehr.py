import bleach
import pandas as pd
import requests

from bs4 import BeautifulSoup
from persiantools.jdatetime import JalaliDate


class MehrCrawler:
    def __init__(self):
        self.base_url = 'https://www.mehrnews.com'

    def crawl_pagination(self, number_of_pages, tp_map):
        month = JalaliDate.today().month
        day = JalaliDate.today().day
        year = JalaliDate.today().year
        news_detail_list = []
        for tp in tp_map.keys():
            for i in range(1, number_of_pages + 1):
                url = f'{self.base_url}/page/archive.xhtml?mn={month}&wide=0&dy={day}&ms=0&pi={i}&yr={year}&tp={tp}'
                resp = requests.get(url).text
                html = BeautifulSoup(resp, 'html.parser')
                news = html.find_all('li', class_='news')
                for desc in enumerate(news):
                    news_detail_list.append({
                        'url': self.base_url + desc[1].find('h3').find('a').get('href'),
                        'topic': tp_map[tp]
                    })

        return news_detail_list

    def crawl_news_detail(self, urls):
        data_list = []
        for url in urls:
            resp = requests.get(url.get('url')).text
            html = BeautifulSoup(resp, 'html.parser')
            text = html.find('div', {'class': 'item-text'})
            cleaned_text = bleach.clean(str(text), tags={}, strip=True)
            text = {'text': cleaned_text,
                    'topic': url.get('topic')
                    }
            data_list.append(text)
            print(url)
        self.create_excel(data_list)

    def create_excel(self, data):
        df = pd.DataFrame.from_dict(data)
        df.to_excel('mehr.xlsx')


if __name__ == '__main__':
    crawler = MehrCrawler()
    "sport, art, economy, politics"
    tp_map = {
        9: 'sports',
        1: 'arts',
        7: 'politics',
        25: 'economy',
    }
    urls = crawler.crawl_pagination(5, tp_map)
    crawler.crawl_news_detail(urls)
    print('[done] Mehr crawler')
