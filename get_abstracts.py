import progressbar
import requests
from bs4 import BeautifulSoup as bs
import time
import json
import concurrent.futures


def progress_bar():
    return progressbar.ProgressBar(maxval=78, widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(marker='0', left='[', right=']'),
        ' (', progressbar.ETA(), ') ',
    ])


abstracts_data = []

# get abstract and title from detail page


def get_abstract_and_title(link):
    res = requests.get(link)
    html_page = bs(res.content, 'html.parser')
    table = html_page.find('table', {'class': 'table table-responsive'})
    table_trs = table.find_all('tr')
    nim = table_trs[2].text.strip()
    if 'K35' in nim or 'K.35' in nim or 'K. 35' in nim or 'K 35' in nim:
        print('==============================')
        print(
            f'Getting abstract from link: {link} \n')
        tdabstrak = table_trs[13].find_all('td')
        tdjudul = table_trs[4].find_all('td')
        abstract = tdabstrak[2].get_text()
        title = tdjudul[2].get_text()

        abstracts_data.append({
            'title': title,
            'abstract': abstract,
        })
    else:
        print('==============================')
        print(
            f'Link skipped: {link} \n')
        time.sleep(1)


def add_id_to_abstract(i):
    abstracts_data[i]['id'] = i + 1


def get_abstracts():
    with open('data/detail_links.json', 'r') as input:
        links_from_json = json.load(input)
        for link in links_from_json:
            get_abstract_and_title(link)
            time.sleep(2)
        for i in range(len(abstracts_data)):
            add_id_to_abstract(i)
        with open('data/abstracts_data.json', 'w') as output:
            output.write(json.dumps(abstracts_data, indent=4, sort_keys=True))


get_abstracts()
