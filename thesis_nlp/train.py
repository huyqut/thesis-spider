from os import path
from random import shuffle

import nltk
from gensim.corpora import WikiCorpus
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import logging
import os.path
import six
import sys
import thesis_nlp.train_database as train_database
import database as news_database
import thesis_logging
from bson.objectid import ObjectId

# Clustering
import hdbscan
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
import pickle
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
import random
# Evaluate
from pathlib import Path
import csv
import math
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import re

# Plotting
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from numpy import dot
from numpy.linalg import norm

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
enwiki_bin_location = 'training/metawiki-20170401-pages-articles.xml.bz2'
enwiki_txt_location = 'training/wiki-documents.txt'
doc2vec_model_location = 'model/doc2vec-model-temp-300.bin'
word2vec_model_location = 'model/word2vec-model.bin'
doc2vec_vectors_location = 'model/doc2vec-vectors.bin'
clustering_model_location = 'model/clustering_model.bin'
doc2vec_dimensions = 300
classifier_model_location = 'model/classifier-model.bin'

train_collection = train_database.train_collection()
news_collection = news_database.news_collection()


# Build the word2vec model from the corpus
# doc2vec.build_vocab(taggedDocuments)

def build_wiki_text():
    i = 0
    output = open(enwiki_txt_location, 'w+', encoding="utf-8")
    wiki = WikiCorpus(enwiki_bin_location, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(b' '.join(text).decode('utf-8') + '\n')
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


def build_wiki_tagged():
    for idx, doc in enumerate(open(enwiki_txt_location, 'r', encoding="utf-8")):
        if idx < 1000000:
            yield TaggedDocument(words=nltk.word_tokenize(doc.lower()), tags=['tag_' + str(idx)])
        else:
            break


def build_news_tagged():
    logger = thesis_logging.get_logger('preprocess')
    latest = ObjectId("5950c589f296532a806e4f31")
    count = 0
    index = 1

    news_collection = train_collection
    duplicated_doc = {}
    while True:
        documents = news_collection.find({'_id': {'$gt': latest}})
        if documents.count() == 0:
            break
        for doc in documents:
            count += 1
            try:
                latest = doc['_id']
                if not doc.get('text'):
                    # print('Ignore', 'Count ' + str(count), 'Id ' + str(doc['id']), str(doc['created_at']),
                    #      doc['reference'])
                    continue
                content = doc['text']
                if len(content) < 100:
                    # logger.info('Ignore small content, Count ' + str(count))
                    continue
                if content not in duplicated_doc:
                    duplicated_doc[content] = True
                    index += 1
                    # logger.info(nltk.word_tokenize(content.lower()))
                    yield TaggedDocument(words=nltk.word_tokenize(content.lower()), tags=['news_' + str(doc['id'])])

            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))


def build_dataset(cluster_datas):
    idx = 0
    for cluster_data in cluster_datas:
        idx += 1
        yield TaggedDocument(words=nltk.word_tokenize(cluster_data.text.lower()),
                             tags=[cluster_data.type + str(idx)])


def train(dataset):
    # build_wiki_text()

    logger.info("Build TaggedDocuments from training docs")

    model = Doc2Vec(size=doc2vec_dimensions, iter=1, window=10, dbow_words=0, dm_concat=1,
                    seed=1337, min_count=5, workers=10, alpha=0.025, min_alpha=0.025)
    logger.info("Build Vocabulary from docs")
    model.build_vocab(build_news_tagged())
    for epoch in range(10):
        logger.info("epoch " + str(epoch))
        model.train(build_news_tagged(), total_examples=model.corpus_count, epochs=1)
        model.save(doc2vec_model_location)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay


def load():
    # from gensim.models import Doc2Vec
    # doc2vec_model_location = 'model/doc2vec-model.bin'
    # model = Doc2Vec.load(doc2vec_model_location)
    # model.docvecs['']  # get vector from tag or index
    # model.docvecs.most_similar(positive=[model.docvecs['']]) # get most similar vector 1
    # model.docvecs.most_similar(positive=[model.infer_vector(words)]) # get most similar vector 2

    doc2vec_model_location = 'temp/doc2vec-model-300.bin'
    # doc2vec_model_location = 'temp/doc2vec-model-400k-300.bin'
    # doc2vec_model_location = 'D:/thesis-spider/thesis_nlp/temp/doc2vec-model-wiki-300.bin'
    logger.info("Loading model...")
    model = Doc2Vec.load(doc2vec_model_location)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    testIds = ['731550159583088641', '549916776734416896', '543030544981032961', '812672127891636224']

    # for id in testIds:
    #     list_similar = model.docvecs.most_similar(positive=[model.docvecs[id]])
    #     for similar in list_similar:
    #         print(similar)
    #         doc = train_collection.find_one({'id': similar[0]})
    #         print(doc['title'])
    #     print('\n')
    text = 'President Donald Trump threatened Tuesday to bring the U.S. government to the brink of a shutdown if needed to pressure Congress into funding the border wall that was a centerpiece of his 2016 campaign.Delivering a warning to Democratic lawmakers who have objected to his plans to construct a wall along the U.S.-Mexico frontier, Trump called them “obstructionists” and said that it was time for the U.S. to crack down on illegal immigration.“If we have to close down our government, we’re building that wall,” Trump told thousands of supporters gathered in Phoenix for a campaign-style rally. “One way or the other, we’re going to get that wall.”Trump’s threats about shutting down the government and ending the North American Free Trade Agreement caused U.S. stock-index futures to pare gains and drop as much as 0.3 percent. Dow futures were down 0.2 percent as were E-Mini Nasdaq 100 futures.Political Capital“Given that Trump’s political capital has diminished, finding support amongst Republicans to approve potentially billions of dollars to fund construction of a controversial wall is likely to prove difficult,” said Rabobank analysts Piotr Matys and Jane Foley.Trump’s comment that he might terminate Nafta at some point caused the yen to strengthen, while the Mexican peso weakened 0.2 percent.“His comments on the NAFTA negotiations once again brings the general direction toward obstructing free trade, and raises concerns over its impact on global trade,” said Hideyuki Ishiguro, a senior strategist at Daiwa Securities Co. in Tokyo.Trump has asked for $1.6 billion to begin construction of the wall, with Congress under pressure to pass some kind of spending bill to keep the government open after Sept. 30.But Republicans in Congress haven’t shown much appetite for fighting to spend potentially billions more on a border barrier either. The funding would add to the deficit at the same time Republicans are trying to figure out how to pay for tax cuts.Debt LimitThe issue could also get wrapped up with legislation to raise the federal government’s debt limit, which needs to be raised between late September and mid-October to avoid a default.One option being considered by GOP leaders is attaching a debt limit measure to the stopgap spending bill that will likely be considered next month. Under that scenario, Trump’s threat to shut down the government over the border wall could entangle the debt ceiling debate.Senate Majority Leader Mitch McConnell said Monday in a speech that he sees "zero chance" that Congress won’t lift the debt limit. Trump’s Treasury secretary, Steven Mnuchin, said at the same event that he will run out of authority to stay under the limit late next month and his priority when Congress returns in early September is ensuring it’s lifted.During his speech, Trump also repeated his call for a historic tax cut. While he provided no details of any planned legislation, he urged congressional Democrats to support it. Democratic senators in states he won should be particularly wary, Trump said. Most Senate Democrats have said they’ll refuse to support any tax legislation that provides a tax cut to the highest earners.“The Democrats are going to find a way to obstruct,” Trump said. If so, he told his supporters, they’ll be preventing Americans from receiving a “massive tax cut.”'
    words = nltk.word_tokenize(text.lower())
    vec = model.infer_vector(words, steps=3, alpha=0.2)

    return model


def get_ner_list(ner_list, doc):
    if len(doc['people']) == 0:
        return
    c = Counter(doc['people'])
    ner = c.most_common(1)[0]

    if ner not in ner_list:
        ner_list[ner] = []
    else:
        ner_list[ner].append(doc['id'])


def plot_data(data_vecs, clusterer):
    pca = PCA(n_components=50)
    fiftyDimVecs = pca.fit_transform(data_vecs)

    tsne = TSNE(n_components=2)
    twoDimVecs = tsne.fit_transform(fiftyDimVecs)

    # centers = pca.fit_transform(clusterer.cluster_centers_)
    # centerDimVecs = tsne.fit_transform(centers)
    # print(centerDimVecs)

    fig, ax = plt.subplots()
    idx = 0

    # for centerDimVec in centerDimVecs:
    #    ax.scatter(centerDimVec[0], centerDimVec[1], c='b')

    colors = plt.cm.get_cmap('hsv', len(clusterer.labels_))

    for twoDimVec in twoDimVecs:
        ax.scatter(twoDimVec[0], twoDimVec[1], c=colors(clusterer.labels_[idx]), s=0.1)
        idx += 1

    plt.show()


def cluster(model):
    idx = 0
    data_vecs = []
    data_tags = []
    for key in model.docvecs.doctags:
        if idx >= 1000:
            break
        data_vecs.append(model.docvecs[key])
        data_tags.append(key)
        idx += 1

    print('Infer doc vectors')
    docs = news_collection.find({})
    duplicate_doc = {}
    people_list = {}
    # for doc in docs:
    #     text = doc['text']
    #     if text in duplicate_doc:
    #         continue
    #     else:
    #         duplicate_doc[text] = True
    #
    #     data_vecs.append(model.infer_vector(doc['text']))
    #     data_tags.append(doc['id'])

    clusterer = clustering(doc_vecs=data_vecs)

    pickle.dump(clusterer, open(clustering_model_location, 'wb'))
    logger.info('Saved clustering model')
    idx = 0
    clusters_list = {}
    for label in clusterer.labels_:
        if label not in clusters_list:
            clusters_list[label] = []
        else:
            clusters_list[label].append(data_tags[idx])
        idx += 1
    print(len(clusters_list))

    for label in clusters_list:
        print(label)
        print(len(clusters_list[label]))
        for article in clusters_list[label]:
            print(article)
            doc = train_collection.find_one({'id': article})
            print(doc['title'])
        print('\n')

        # print('Inertia: ', clusterer.inertia_)
        # plot_data(data_vecs, clusterer)


def clustering(doc_vecs, k):
    def sqrt(list):
        return [math.sqrt(math.fabs(i)) * math.fabs(i) / i for i in list]

    data_vecs = doc_vecs
    print('Normalize vectors...')
    data_vecs = preprocessing.normalize(doc_vecs, norm='l2')

    # vecs = []
    # for vec in data_vecs:
    #     vec = sqrt(vec)
    #     vecs.append(vec)
    #     # print(max(vec), min(vec))
    #
    # data_vecs = preprocessing.normalize(vecs, norm='l2')
    # print(data_vecs)

    # pca = PCA(n_components=15)
    # data_vecs = pca.fit_transform(data_vecs)

    print('Start clustering...')
    distance = pairwise_distances(data_vecs, metric='l2')
    # clusterer = Birch(branching_factor=50, n_clusters=5, threshold=5, compute_labels = True)
    # clusterer = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=5, prediction_data=True)
    clusterer = KMeans(n_clusters=k, max_iter=1000, n_init=1000)
    # clusterer =  AgglomerativeClustering(n_clusters=5, linkage="ward", affinity='euclidean')

    clusterer.fit(data_vecs)

    # print('Generate prediction data')
    # clusterer.generate_prediction_data()
    return clusterer


class ClusterData:
    def __init__(self, type, path, vec, text, split='test'):
        self.type = type
        self.path = path
        self.vec = vec
        self.text = text
        self.split = split



def get_reuters():
    path = './datasets/reuters/r8-test-all-terms.txt'
    doc_vecs = []
    cluster_datas = []
    with open(path, 'r') as doc_file:
        for line in doc_file:
            segs = line.split('\t')
            cluster_datas.append(ClusterData(segs[0], '', 0, segs[1]))

    return cluster_datas


def get_bbc():
    p = Path('./datasets/bbc')
    cluster_datas = []
    for f in p.iterdir():
        if f.is_dir():
            list_docs_iter = Path(str(f))
            list_docs = [x for x in list_docs_iter.iterdir()]
            random.shuffle(list_docs)
            size = len(list_docs) / 2
            print(str(f), size * 2)
            for doc_path in list_docs:
                with open(doc_path, 'r') as doc_file:
                    data = doc_file.read()
                    if size < 1:
                        cluster_datas.append(ClusterData(f.name, doc_path, 0, data, 'test'))
                    else:
                        cluster_datas.append(ClusterData(f.name, doc_path, 0, data, 'train'))
                        size -= 1
    return cluster_datas


def get_news20():
    # 20ng-test-no-short  20ng-test-all-terms.txt
    path = './datasets/news20/20ng-test-all-terms.txt'
    cluster_datas = []
    with open(path, 'r') as doc_file:
        for line in doc_file:
            segs = line.split('\t')
            cluster_datas.append(ClusterData(segs[0], '', 0, segs[1], 'test'))

    path = './datasets/news20/20ng-train-all-terms.txt'
    with open(path, 'r') as doc_file:
        for line in doc_file:
            segs = line.split('\t')
            cluster_datas.append(ClusterData(segs[0], '', 0, segs[1], 'train'))
    return cluster_datas


def get_amazon_6():
    import random
    p = Path('./datasets/amazon_6/')
    cluster_datas = []
    for f in p.iterdir():
        if f.is_dir():
            count = 0
            list_docs = Path(str(f))
            for doc_path in list_docs.iterdir():
                if count > 1500:
                    break
                with open(doc_path, 'r') as doc_file:
                    xml = doc_file.read().strip()
                    matches = re.findall('Content": "[^"]*', xml)
                    for review in matches:
                        review = review.replace('Content": "', '')
                        if len(review) < 200:
                            continue
                        cluster_datas.append(ClusterData(f.name, doc_path, 0, review))
                        count += 1
            print(str(f), count)
    print(len(cluster_datas))
    return cluster_datas


def get_amazon_4():
    p = Path('./datasets/amazon_4/')
    cluster_datas = []
    for f in p.iterdir():
        if f.is_dir():
            list_docs = Path(str(f))
            print(str(f))
            for doc_path in list_docs.iterdir():
                with open(doc_path, 'r') as doc_file:
                    list_lines = doc_file.readlines()
                    size = len(list_lines) / 2
                    random.shuffle(list_lines)
                    for line in list_lines:
                        data = line
                        if size < 1:
                            cluster_datas.append(ClusterData(f.name, doc_path, 0, data, 'test'))
                        else:
                            cluster_datas.append(ClusterData(f.name, doc_path, 0, data, 'train'))
                            size -= 1
            print(len(cluster_datas))
    return cluster_datas


def frequency_word(cluster_datas):
    import statistics
    word_array = {}
    fre_array = {}
    for cluster_data in cluster_datas:
        words = nltk.word_tokenize(cluster_data.text.lower())
        for word in words:
            if word not in word_array:
                word_array[word] = 0
            word_array[word] += 1

    # fre_array  key: frequency, value: word count
    for value in word_array.values():
        if value not in fre_array:
            fre_array[value] = 0
        fre_array[value] += 1
    max_fre = max(fre_array.keys())
    fre_data = np.zeros(max_fre + 1)
    for key in fre_array.keys():
        fre_data[key] = fre_array[key]

    # fre_data = [(math.log10(x)) if x > 0 else x for x in fre_data]
    # print(len(fre_data))
    # print(fre_data)
    # re_data = [10, 11, 10, 12, 15]
    array_data = [x for x in word_array.values()]
    n, bins, patche = plt.hist(array_data, log=0, bins=300, density=True)

    # add a 'best fit' line
    y = mlab.normpdf(bins, statistics.mean(array_data), statistics.stdev(array_data))
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Frequency')
    plt.ylabel('Word Count')
    plt.title('BBC')
    # plt.axis([0, 100, 0, 1000])
    # plt.grid(True)
    # plt.ylim([0, 1000])

    plt.show()


def build_docvecs(model, cluster_datas):
    idx = 0
    doc_vecs = []
    labels = []
    for cluster_data in cluster_datas:
        words = nltk.word_tokenize(cluster_data.text.lower())
        vec = model.infer_vector(words, steps=3, alpha=0.2)
        doc_vecs.append(vec)
        labels.append(cluster_data.type)
        cluster_datas[idx].vec = vec
        idx += 1
    return doc_vecs, labels, cluster_datas


def evaluate_retrain():
    doc2vec_model_location = 'model/doc2vec-model-bbc-300.bin'
    # cluster_datas = get_bbc()
    import datetime
    # dataset = build_dataset(train_set)
    # train(dataset)
    model = load()
    print('Evaluate with old model, k = 19, step = 3, al = 0.2')
    for i in range(0, 15):
        cluster_datas = get_bbc()
        train_set = [cluster_data for cluster_data in cluster_datas if cluster_data.split == 'train']
        test_set = [cluster_data for cluster_data in cluster_datas if cluster_data.split == 'test']
        doc_vecs_train, labels_train, cluster_datas_train = build_docvecs(model, train_set)

        print('Normalize train vectors...', datetime.datetime.now().time())
        doc_vecs_train = preprocessing.normalize(doc_vecs_train, norm='l2')

        print('Training data...', len(labels_train), datetime.datetime.now().time())
        neigh = KNeighborsClassifier(n_neighbors=19)
        neigh.fit(doc_vecs_train, labels_train)

        doc_vecs_test, labels_test, cluster_datas_test = build_docvecs(model, test_set)

        print('Normalize test vectors...', datetime.datetime.now().time())
        doc_vecs_test = preprocessing.normalize(doc_vecs_test, norm='l2')

        print('Testing...', len(labels_test), datetime.datetime.now().time())
        result = neigh.predict(doc_vecs_test)

        # clusterer = KMeans(n_clusters=5, max_iter=1000, n_init=1000)
        # clusterer.fit_predict(doc_vecs_test, labels_test)
        # result = clusterer.labels_
        # print(labels_test)
        # print(result)
        corrects = sum((result) == labels_test)
        errors = len(cluster_datas_test) - corrects
        error_rate = float(errors) / len(cluster_datas_test)
        print('error', errors, '/', len(cluster_datas_test), datetime.datetime.now().time())
        print('error rate:', error_rate)
        print('Correct:', corrects / (len(cluster_datas_test)))
        print('\n\n')


def evaluate(model):
    import statistics

    doc_vecs = []
    cluster_datas = []

    cluster_datas = get_amazon_4()
    frequency_word(cluster_datas)
    print('Infer vectors...')
    idx = 0
    count = 0
    words_count = []
    words_list = set()
    for cluster_data in cluster_datas:
        words = nltk.word_tokenize(cluster_data.text.lower())
        count += len(words)
        for word in words:
            words_list.add(word)
        words_count.append(len(words))
        vec = model.infer_vector(words)
        doc_vecs.append(vec)
        cluster_datas[idx].vec = vec
        idx += 1

    print(count)
    print('docs', len(words_count))
    print('Vob', len(words_list))
    print('Mean:', count / len(words_count), statistics.mean(words_count))
    print('Stdev:', statistics.stdev(words_count))

    clusterer = clustering(doc_vecs, 4)
    labels_list = clusterer.labels_

    # from collections import Counter
    # clusterer_list = []
    # for i in range(5):
    #     doc_vecs = []
    #     idx = 0
    #     for cluster_data in cluster_datas:
    #         vec = model.infer_vector(nltk.word_tokenize(cluster_data.text))
    #         doc_vecs.append(vec)
    #         cluster_datas[idx].vec = vec
    #         idx += 1
    #     clusterer_list.append(clustering(doc_vecs))
    #
    # for clusterer in clusterer_list:
    #         print(clusterer.labels_)

    # labels_list = []
    # for i in range(len(cluster_datas)):
    #     temp_list = []
    #     for clusterer in clusterer_list:
    #         temp_list.append(clusterer.labels_[i])
    #     common_data = Counter(temp_list)
    #     labels_list.append(common_data.most_common(1)[0][0])

    print('Relabels: ', labels_list)
    # K nearest neighbors
    # neigh_top_10 = NearestNeighbors(11, metric='euclidean')
    # neigh_label = NearestNeighbors(3, metric='euclidean')
    # neigh_top_10.fit(doc_vecs)
    # neigh_label.fit(clusterer.cluster_centers_)

    idx = 0
    clusters_list = {}
    for label in labels_list:
        if label not in clusters_list:
            clusters_list[label] = []
        clusters_list[label].append(cluster_datas[idx])
        idx += 1
    print(len(clusters_list))

    cluster_sum = 0
    doc_sum = 0
    max_key_list = {}
    with open('cluster.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(
            ['Document', 'Cluster name', 'labeled', 'Distances 10 neighbors', 'Distances 3 labels', 'Text'])
        for label in clusters_list:
            dict_docs = {}
            dict_doc_datas = {}
            idx = 0
            for article in clusters_list[label]:
                if article.type not in dict_docs:
                    dict_docs[article.type] = 0
                    dict_doc_datas[article.type] = []
                dict_docs[article.type] += 1
                dict_doc_datas[article.type].append(article)
                doc_sum += 1
                idx += 1

            max_key = max(dict_docs.keys(), key=(lambda key: dict_docs[key]))
            if max_key not in max_key_list:
                max_key_list[max_key] = 0
            max_key_list[max_key] = dict_docs[max_key] if max_key_list[max_key] < dict_docs[max_key] else \
                max_key_list[max_key]

            print('Cluster size:', len(clusters_list[label]))
            print(dict_docs)

            for key in dict_docs:
                if key != max_key:
                    # print('- ' + key + ': ')
                    for article in dict_doc_datas[key]:
                        # top_10 = neigh_top_10.kneighbors([article.vec])[0][0][1:]
                        # top_labels = neigh_label.kneighbors([article.vec])[0]
                        # spamwriter.writerow(
                        #    [article.path, max_key, 'True' if article.type == max_key else 'False', top_10, top_labels, article.text])
                        print(article.path, end=" ")
                        # print('  + Distances between top 10 nearest neighbors', top_10)
                        # print('  + Distances between top 3 nearest labels', top_labels)
                        # print()
            print()
        for max_key in max_key_list.keys():
            cluster_sum += max_key_list[max_key]

        print('\n')

    print('Sum= ', cluster_sum, '/', doc_sum, '= ', cluster_sum / doc_sum)


def eblow(data, n):
    from scipy.spatial.distance import cdist, pdist
    df = np.array(data)
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(2, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d ** 2) for d in dist]
    tss = sum(pdist(df) ** 2) / df.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()


def extract_amazon_4():
    p = Path('./datasets/amazon_4/')
    for f in p.iterdir():
        if f.is_dir():
            print(str(f))
            list_docs = Path(str(f))
            new_path = str(f) + '_extracted'
            file = open(new_path, 'w')
            for doc_path in list_docs.iterdir():
                with open(doc_path, 'r') as doc_file:
                    xml = doc_file.read().strip()
                    matches = re.findall('<review_text>[^</]*</review_text>', xml)
                    print(len(matches))

                    for review in matches:
                        review = review.replace('<review_text>', '')
                        review = review.replace('</review_text>', '')
                        review = review.replace('\n', '')
                        file.write(review + '\n')

            file.close()
            file = open(new_path, 'r')
            print(len(file.readlines()))
            file.close()


if __name__ == '__main__':
    train({})
    #load()
    # evaluate_retrain()
