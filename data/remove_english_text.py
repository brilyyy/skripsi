import json
from langdetect import detect


def remove_english_text():
    with open('./data/documents.json') as d:
        docs = list(json.load(d))

    for i, doc in enumerate(docs):
        if (detect(doc['abstrak']) == 'en'):
            docs.pop(i)
        print(doc['id'], detect(doc['abstrak']))

    with open('./data/documents-id.json', "w") as json_file:
        json.dump(docs, json_file, indent=4, separators=(",", ": "))


if __name__ == '__main__':
    remove_english_text()