import json, re, nltk, math
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np
from pandas import DataFrame
from sklearn_extra.cluster import KMedoids
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def log(msg):
    file = open('./kmedoids/kmedoids_log.txt', 'a')
    file.write(f'{msg} \n')
    file.close()


# Case Folding
with open('data/documents-id.json') as abstracts_json:
    abstracts = json.load(abstracts_json)

for i, abstract in enumerate(abstracts):
    answer = re.sub('[^a-z]+', ' ', abstract['abstrak'].casefold())
    abstracts[i]['abstrak'] = answer

with open('kmedoids/1_case_folded.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

log('Case folding done')

# Stemming
stemmer = MPStemmer()

with open('kmedoids/1_case_folded.json') as case_folded_json:
    abstracts = json.load(case_folded_json)

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

for i, abstract in enumerate(abstracts):
    stemmed = stemmer.stem_kalimat(abstract['abstrak'])
    abstracts[i]['abstrak'] = stemmed

with open('kmedoids/2_stemmed_abstracts.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

# Filtering
with open('kmedoids/2_stemmed_abstracts.json') as json_file:
    abstracts = json.load(json_file)

additional_stopwords = []

stopwords = StopWordRemoverFactory().get_stop_words() + additional_stopwords

for i, abstract in enumerate(abstracts):
    words = abstract['abstrak'].split()
    filtered = ''
    for word in words:
        if word not in stopwords:
            filtered += word + ' '
    abstracts[i]['abstrak'] = filtered

with open('kmedoids/3_filtered.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

# Tokenize
nltk.download('punkt')

with open('kmedoids/3_filtered.json') as json_file:
    abstracts = json.load(json_file)

for i, abstract in enumerate(abstracts):
    frequency = nltk.FreqDist(nltk.word_tokenize(abstract['abstrak']))
    abstracts[i]['frequency'] = frequency

with open('kmedoids/4_tokenized.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

# TF
with open('kmedoids/4_tokenized.json') as json_file:
    abstracts = json.load(json_file)

for i, abstract in enumerate(abstracts):
    result = {}
    bagOfWordsCount = len(abstract['abstrak'].split())
    for word, count in abstract['frequency'].items():
        if (count / float(bagOfWordsCount)):
            result[word] = count / float(bagOfWordsCount)
    abstracts[i]['frequency'] = result

with open('kmedoids/5_tf.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

# IDF

with open('kmedoids/4_tokenized.json') as json_file:
    abstracts = json.load(json_file)

abstract_length = len(abstracts)

for i, abstract in enumerate(abstracts):
    idfDict = dict.fromkeys(abstract['frequency'].keys(), 0)
    result = {}
    for key in idfDict.keys():
        for abstract in abstracts:
            if key in abstract['frequency']:
                idfDict[key] += 1
    for word, val in idfDict.items():
        result[word] = math.log10(abstract_length / val)
    abstracts[i]['frequency'] = result

with open('kmedoids/6_idf.json', 'w') as outfile:
    outfile.write(json.dumps(abstracts, sort_keys=True, indent=4))

# TF-IDF
with open('kmedoids/6_idf.json') as idf_json:
    idfs = json.load(idf_json)

with open('kmedoids/5_tf.json') as tf_json:
    tfs = json.load(tf_json)

for i, idf in enumerate(idfs):
    tfidf = {}
    maxtfidf = 0
    for word, val in idf['frequency'].items():
        if tfs[i]['frequency'].get(word) is not None:
            tfidf[word] = val * float(tfs[i]['frequency'].get(word))
            if maxtfidf < tfidf[word]:
                maxtfidf = tfidf[word]
                maxtfidfword = word
    idfs[i]['frequency'] = tfidf
    idfs[i]['maxtfidfword'] = maxtfidfword

with open('kmedoids/7_tfidf.json', 'w') as json_file:
    json_file.write(json.dumps(idfs, sort_keys=True, indent=4))

# Creating dictionary
with open('kmedoids/7_tfidf.json') as json_file:
    documents = json.load(json_file)

words = []

for i, document in enumerate(documents):
    for word, val in document['frequency'].items():
        if word not in words:
            words.append(word)

words_dict = dict(zip(range(len(words)), words))

with open('kmedoids/8_dictionary.json', 'w') as outfile:
    outfile.write(json.dumps(words_dict, sort_keys=True, indent=4))

# Replace word with dictionary id
with open('kmedoids/8_dictionary.json') as json_file:
    words = json.load(json_file)
with open('kmedoids/7_tfidf.json') as json_file:
    documents = json.load(json_file)


# define function to get key from dictionary
def get_key(val):
    for key, value in words.items():
        if val == value:
            return key


for i, document in enumerate(documents):
    result = {}
    for word, val in document['frequency'].items():
        if word in words.values():
            result[get_key(word)] = val
    documents[i]['frequency'] = result

with open('kmedoids/9_dictionarized.json', 'w') as outfile:
    outfile.write(json.dumps(documents, sort_keys=True, indent=4))

with open('kmedoids/8_dictionary.json') as json_file:
    dictionary = json.load(json_file)

with open('kmedoids/9_dictionarized.json') as json_file:
    documents = json.load(json_file)

start_time = time.time()

array = []
for i, document in enumerate(documents):
    new = []
    for key in dictionary.keys():
        if not document['frequency'].get(key) is None:
            new.append(document['frequency'].get(key))
        else:
            new.append(0)
    array.append(new)

X = np.array(array)
log(X)
titles = []

for document in documents:
    titles.append(document['judul'])

kmedoids = KMedoids(n_clusters=5).fit(X)

print("--- %s seconds ---" % (time.time() - start_time))

kmedoids_dataframe = DataFrame({
    'Cluster': kmedoids.labels_.tolist(),
    'Title': titles,
})

sorted_data_frame = kmedoids_dataframe.sort_values(by=['Cluster'])
sorted_data_frame.to_excel('kmedoids/result.xlsx', index=False)
