import bleach
import pandas as pd
import requests

from bs4 import BeautifulSoup
from persiantools.jdatetime import JalaliDate


class HamshahriCrawler:
    def __init__(self):
        self.base_url = 'https://www.hamshahrionline.ir'

    def crawl_pagination(self, number_of_pages, tp_map):
        month = JalaliDate.today().month
        day = JalaliDate.today().day

        news_detail_list = []
        for tp in tp_map.keys():
            for i in range(1, number_of_pages + 1):
                url = f'{self.base_url}/page/archive.xhtml?mn={month}&wide=0&dy={day}&ms=0&pi={i}&tp={tp}'
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
        df.to_excel('hamshahri.xlsx')


if __name__ == '__main__':
    crawler = HamshahriCrawler()
    "sport, art, economy, politics"
    tp_map = {
        9: 'sports',
        26: 'arts',
        10: 'economy',
        6: 'politics',
    }
    urls = crawler.crawl_pagination(5, tp_map)
    crawler.crawl_news_detail(urls)
    print('[done] Hamshahri crawler')
