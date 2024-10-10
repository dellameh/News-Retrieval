import csv
import string
import pandas as pd
import hazm
import json
import numpy as np
from hazm import stopwords_list
from scipy.sparse import csr_array
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

normalizer = hazm.Normalizer()
word_tokenizer = hazm.WordTokenizer()
stemmer = hazm.Stemmer()
tf_vec = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(2, 2))


def Jaccard_Similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


class IR:

    def __init__(self, data, gram=1, n=0):
        self.data = data
        self.n = n
        self.gram = gram

        self.text = (self.data['text']).to_string()
        self.text_list = (self.data['text']).to_list()
        self.topic_list = (self.data['topic']).to_list()
        self.tt = []
        # normalize & tokenize
        for i in self.text_list:
            i = normalizer.normalize(i)
            self.tt.append(i)
        self.text_list = self.tt

        if self.n == 1:
            self.tt = []
            for i in self.text_list:
                i = stemmer.stem(i)
                self.tt.append(i)
            self.text_list = self.tt

        elif self.n == 2:
            self.tt = []

            for i in self.text_list:
                clean_words = []

                for word in i.split():
                    if word not in stopwords_list():
                        clean_words.append(word)

                self.tt.append(' '.join(clean_words))
                # print(' '.join(clean_words))

            self.text_list = self.tt

        elif self.n == 3:
            self.tt = []
            for i in self.text_list:
                i = stemmer.stem(i)
                self.tt.append(i)
            self.text_list = self.tt
            self.tt = []
            for i in self.text_list:
                clean_words = []

                for word in i.split():
                    if word not in stopwords_list():
                        clean_words.append(word)

                self.tt.append(' '.join(clean_words))
            self.text_list = self.tt
        self.tt = []
        for i in self.text_list:
            i = word_tokenizer.tokenize(i)
            self.tt.append(i)
        self.token = self.tt

        # ----->  tfidf Unigram and Bigram
        if self.gram == 1:
            # get tf-df values
            self.result = tf_vec.fit_transform(self.text_list)
            # print(result)

            # get idf values
            """print('\nidf values:')
            for ele1, ele2 in zip(tf_vec.get_feature_names_out(), tf_vec.idf_):
                print(ele1, ':', ele2)"""
            # get indexing
            """print('\nWord indexes:')
            print(tf_vec.vocabulary_)
            # display tf-idf values
            print('\ntf-idf value:')
            print(self.result)"""

        elif self.gram == 2:

            vectorizer = CountVectorizer(ngram_range=(2, 2))
            X1 = vectorizer.fit_transform(self.text_list)
            features = (vectorizer.get_feature_names_out())

            # Applying TFIDF 
            # You can still get n-grams here 
            self.vectorizer = TfidfVectorizer(ngram_range=(2, 2))
            self.result = self.vectorizer.fit_transform(self.text_list)
            
            """sums = self.result.sum(axis=0)
            data1 = [] 
            for col, term in enumerate(features): 
                data1.append( (term, sums[0, col] )) 
            ranking = pd.DataFrame(data1, columns = ['term', 'tf-idf']) 
            words = (ranking.sort_values('tf-idf', ascending = False)) 
            print ("\n\nWords : \n", words.head(10))"""

        elif self.gram == 0:

            self.model = SentenceTransformer('sentence-transformers/LaBSE')
            self.result = self.model.encode(self.text_list)

    # dictionary and counting tf
    def dictionary(self):
        word_count = {}
        # print(self.token)
        for j in self.token:
            for word in j:
                if word not in word_count:
                    word_count[word] = 1
                elif word in word_count:
                    word_count[word] += 1

        sorteddict = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        converted_dict = dict(sorteddict)
        with open('convert.txt', 'w', encoding='utf-8') as convert_file:
            convert_file.write(json.dumps(converted_dict, ensure_ascii=False))
        return converted_dict

    # ----->   bar chart 50 first frequent words
    def barChart(self):
        converted_dict = self.dictionary()
        names = list(converted_dict.keys())[0:50]
        values = list(converted_dict.values())[0:50]
        plt.bar(range(50), values, tick_label=names)
        plt.show()

        # -----> Query

    def query(self):
        query = pd.read_csv("query.csv")
        qtext = (query['text']).to_string()
        self.qtext_list = (query['text']).to_list()
        self.qtopic_list = (query['topic']).to_list()
        self.qt = []
        # normalize & tokenize
        for i in self.qtext_list:
            i = normalizer.normalize(i)
            self.qt.append(i)
        self.qtext_list = self.qt
        #  preprocessing
        if self.n == 1:
            self.qt = []
            for i in self.qtext_list:
                i = stemmer.stem(i)
                self.qt.append(i)
            self.qtext_list = self.qt

        elif self.n == 2:
            self.qt = []
            for i in self.qtext_list:
                clean_words = []

                for word in i.split():
                    if word not in stopwords_list():
                        clean_words.append(word)

                self.qt.append(' '.join(clean_words))
            self.qtext_list = self.qt

        elif self.n == 3:
            self.qt = []
            for i in self.qtext_list:
                i = stemmer.stem(i)
                self.qt.append(i)
            self.qtext_list = self.qt
            self.qt = []
            for i in self.qtext_list:
                clean_words = []

                for word in i.split():
                    if word not in stopwords_list():
                        clean_words.append(word)

                self.qt.append(' '.join(clean_words))
            self.qtext_list = self.qt

        # for i in self.qtext_list:
        #     i=word_tokenizer.tokenize(i)
        #     self.qt.append(i)
        # self.qtext_list=self.qt

        # vector
        if self.gram == 1:
            # uni query
            query = tf_vec.transform(self.qtext_list)
        elif self.gram == 2:
            # bi query
            query = self.vectorizer.transform(self.qtext_list)
        elif self.gram == 0:
            # labse
            query = self.model.encode(self.qtext_list)
        return query
        # print(q2)

    # -----> cosine similarity
    def cosine(self):
        query = self.query()
        li = []
        # cosine=cosine_similarity(query_vec,result)
        cosine = cosine_similarity(query, self.result)
        return cosine

    # -----> Jaccard similarity
    def jaccard(self):
        self.query()
        query = self.qtext_list
        Jaccard = []
        for j in query:
            J = []
            for i in range(len(self.text_list)):
                J.append(Jaccard_Similarity(self.text_list[i], j))
            # print(query)
            Jaccard.append(J)
        # print(Jaccard)
        return Jaccard

    # -----> precision and recall 
    def preRec(self, simNum=1, k=1):
        if simNum == 1:
            res = self.cosine()
        else:
            res = self.jaccard()
        relevant = []
        topicBase = []
        simBase = []
        li = []
        for i in range(len(res)):
            # i=0-50
            li2 = []
            li3 = []
            # li4 = []
            for j in range(len(res[0])):
                # j=0-531
                if res[i][j] > 0 and self.topic_list[j] == self.qtopic_list[i]:
                    li2.append((j, res[i][j]))

                if self.topic_list[j] == self.qtopic_list[i]:
                    li3.append((j, res[i][j]))

                # if res[i][j] > 0:
                #     li4.append((j, res[i][j]))

            if len(li2) == 0:
                li2.append((0, 0))

            if len(li3) == 0:
                li3.append((0, 0))

            # if len(li4) == 0:
            #     li3.append((0, 0))

            relevant.append(len(li2))
            topicBase.append(len(li3))

            li2 = sorted(li2, key=lambda x: x[1], reverse=True)
            li.append(li2)
        best = []

        for i in li:
            # print(i[0])
            best.append(i[0:k])

        # precision & recall
        recall = []
        precision = []
        for i in range(len(best)):
            count = 0
            for j in range(len(best[i])):
                if self.topic_list[best[i][j][0]] == self.qtopic_list[i]:
                    if relevant[i] > k:
                        res = k/k

                    else:
                        #left = k - relevant[i]
                        res=relevant [i]/k

                        # li3[0:left]

                elif best[i][j][0] == 0:
                    res='not found'
                    continue
                else:
                    res=0
                    continue

            precision.append(res)
        for i in range(len(best)):
            res = 0
            for j in range(len(best[i])):
                if best[i][j][0] == 0:
                    res = 'not found'
                    continue
                if k <= relevant[i]:
                    res = k / relevant[i]
                elif k > relevant[i]:
                    res = 1
            recall.append(res)

        print(k, "_best:\n", best)
        print("____________________________________ \n")
        print("precision:\n", precision)
        print("____________________________________ \n")
        print("recall\n", recall)



data = pd.read_csv("corpus.csv")
c1 = IR(data, 2, 1)

c1.preRec(1, 1)

#c1.barChart()