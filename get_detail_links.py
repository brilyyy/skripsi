# Get all links to the detail page from digilib.uns.ac.id
import requests
from bs4 import BeautifulSoup as bs
import time
import json
import concurrent.futures

links = []

abstracts_data = []

pages = 1812

MAX_THREADS = 8
def get_link(page):
  time.sleep(1)
  print(f'Getting link from page {page}')
  res = requests.get(f'https://digilib.uns.ac.id/dokumen/fakultas/7/Fak-KIP/{page}')
  html_page = bs(res.content, 'html.parser')
  document_cards = html_page.select(
      '#digilib-body > div > div > div.col-md-8 > div.mb-5')
  for card in document_cards:
    document_type = card.select(
        '.dokumen-search-body .detail div:nth-child(2)')[0].text.strip().lower()
    if(document_type == 'skripsi'):
      anchor = card.find('a')
      if 'https://digilib.uns.ac.id/dokumen/detail/' in anchor.get('href'):
          links.append(anchor['href'])

def get_detail_links():
  with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
      executor.map(get_link, range(1, pages + 1))
  with open('detail_links.json', 'w') as output:
    output.write(json.dumps(links, indent=4, sort_keys=True))

get_detail_links()