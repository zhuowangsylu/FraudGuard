# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:15:14 2019

@author: hasee
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics.distance import jaccard_distance
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
from textblob import TextBlob
import numpy as np
import nltk
import math

PP_1_Word = set()
with open('1PP-words.txt', 'r') as f:
    for w in f:
        PP_1_Word.add(w.strip())
f.close()
#print(PP_1_Word)

PP_2_Word = set()
with open('2PP-words.txt', 'r') as f:
    for w in f:
        PP_2_Word.add(w.strip())
f.close()
#print(PP_2_Word)

StopWords = set(stopwords.words('english'))

def WordCount(reviewtext, stop_words = False):
    line = word_tokenize(reviewtext)
    
    if stop_words:
        line = [word for word in line if word not in StopWords]
    
    countWord = len(line)
    countPP_1 = 0   #第一人称代词
    countPP_2 = 0   #第二人称代词
    countAllCapital = 0     #全大写字母单词个数
    for li in line:
        if li.lower() in PP_1_Word: countPP_1 += 1
        if li.lower() in PP_2_Word: countPP_2 += 1
        if li.isupper(): countAllCapital += 1
                     
    countPro = countPP_1 + countPP_2
    ratioPP1, ratioPP2 = ((float(countPP_1)/countPro), (float(countPP_2)/countPro)) if countPro > 0 else (0.0, 0.0)
    ratioAllCapital = (float(countAllCapital)/countWord) if countWord > 0 else 0.0      
    return countWord, ratioPP1, ratioPP2, ratioAllCapital    

def WordLower(reviewtext, stop_words=False):
    line = word_tokenize(reviewtext)
    if stop_words:
        line = [word.lower() for word in line if word not in StopWords]
    
    return set(line)

#二元语法相似度
def bigram_distance(text1, text2, stop_words=False):
    word1list = word_tokenize(text1)
    word2list = word_tokenize(text2)
    
    if stop_words:
        word1list = [word.lower() for word in word1list if word not in StopWords]
        word2list = [word.lower() for word in word2list if word not in StopWords]
    
    #bigram考虑匹配开头和结束，所以使用pad_right和pad_left    
    text1_bigrams = nltk.bigrams(word1list, pad_right=True,pad_left=True)
    text2_bigrams = nltk.bigrams(word2list, pad_right=True,pad_left=True)
    #交集的长度
    distance = len(set(text1_bigrams).intersection(set(text2_bigrams)))
    
    return distance

'''
Jaccard相似度 距离越大数值越大 从0-1之间， 0.0表示距离很近，1.0表示距离很远
masi距离度量是jaccard相似度的加权版本，0.0表示距离很近，或无相似，数值分布要比Jaccard小一些
当集合之间存在部分重叠时，通过调整得分来生成小于jaccard距离值。
'''

def JaccardSimAndMasiDis(text1, text2, stop_words=False):
    word1list = word_tokenize(text1)
    word2list = word_tokenize(text2)
    
    if stop_words:
        word1list = [word.lower() for word in word1list if word not in StopWords]
        word2list = [word.lower() for word in word2list if word not in StopWords]
    
    word1set = set(word1list)
    word2set = set(word2list)
    
    return 1 - jaccard_distance(word1set, word2set)#, 1 - masi_distance(word1set, word2set)

def JaccardSim(text1, text2, stop_words=False):
    word1list = word_tokenize(text1)
    word2list = word_tokenize(text2)
    
    if stop_words:
        word1list = [word.lower() for word in word1list if word not in StopWords]
        word2list = [word.lower() for word in word2list if word not in StopWords]
    
    word1set = set(word1list)
    word2set = set(word2list)
    
    return len(word1set.intersection(word2set))/len(word1set.union(word2set))

#余弦相似度
def cosine_Distance(text1, text2, stop_words=False, mostCommon=20):
    word1list = word_tokenize(text1)
    word2list = word_tokenize(text2)
    
    if stop_words:
        word1list = [word.lower() for word in word1list if word not in StopWords]
        word2list = [word.lower() for word in word2list if word not in StopWords]
    
    fdist = FreqDist(word1list + word2list)
    topFdist = fdist.most_common(mostCommon)
    text1_vector = []
    text2_vector = []
    for t in topFdist:
        tCount1 = 0
        tCount2 = 0
        
        for w in word1list:
            if t[0] == w:   tCount1 += 1
        text1_vector.append(tCount1)
        
        for w in word2list:
            if t[0] == w:   tCount2 += 1
        text2_vector.append(tCount2)
    
    if math.isnan(cosine_distance(text1_vector,text2_vector)):
        return 0.0
    else:
        return 1 - cosine_distance(text1_vector,text2_vector)

def cosine(tv1, tv2):
    return np.dot(tv1, tv2) / (
                math.sqrt(np.dot(tv1, tv1)) * math.sqrt(np.dot(tv2, tv2)))

def cosine_Distance2(text1, text2, stop_words=False):
    w1 = word_tokenize(text1)
    w2 = word_tokenize(text2)
	
    if stop_words:
        w1 = [word.lower() for word in w1 if word not in StopWords]
        w2 = [word.lower() for word in w2 if word not in StopWords]
	
    wordSet = list(set(w1 + w2))
    tv1 = []
    tv2 = []

    for w in wordSet:
        tc1 = 0
        tc2 = 0
        while w in w1:
            w1.remove(w)
            tc1 += 1
        tv1.append(tc1)
        while w in w2:
            w2.remove(w)
            tc2 += 1
        tv2.append(tc2)
    
    try:
        if np.dot(tv1, tv1) == 0.0 or np.dot(tv2, tv2) == 0.0:
            raise RuntimeWarning
        
    except RuntimeWarning:
        print("VECTOR is Null")
        print("vector1: ", tv1)
        print("vector2: ", tv2)
        return 0.0
    
    else:
        return cosine(tv1, tv2)
    
def SentimentAnalyze(text):
    blob = TextBlob(text)
    
    return blob.sentiment[0], blob.sentiment[1]