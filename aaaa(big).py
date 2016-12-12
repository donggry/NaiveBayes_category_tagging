# coding: utf-8
from __future__ import print_function
import numpy as np
from operator import itemgetter

import string
import numpy as np
import textmining
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords
wnl = WordNetLemmatizer()

StopWords = set(stopwords.words("english"))
StopWords.update(('and', 'may', 'a', 'use', 'us','so', 'your', 'this', 'when', 'it', 'many', 'can', 'set', 'cant',
                            '’','®','™','®','with','º','in','to','v', 'w\xc2\xb0','yes', 'not', 'no', 'these','keep','enough','use','x','w','d','p','n'))

g=open("tfidf1.txt","w")
t=open("tfidf11.txt","w")
vocab=[]

doc_dic=[]
for a in range(11):###################################small category change!!!!!!!!
    doc_dic.append([])
"""

category_dic2={'Electronics / Musical Instruments': 54, 'Auto & Home Improvement / Automotive': 40, 'Electronics / Portable Audio': 14, 'Health & Beauty / Cosmetics': 0, 'Health & Beauty / Sexual Wellness': 35, 'Health & Beauty / Skin Care': 42, "Women's Fashion / Shoes": 24, 'Electronics / Computers & Tablets': 13, 'Health & Beauty / Bath & Body': 12, 'Sports & Outdoors / Golf': 20, 'Electronics / Camera, Video & Surveillance': 46, "Jewelry & Watches / Men's Jewelry": 16, 'Electronics / Software': 81, 'For the Home / Kitchen & Dining': 63, 'Electronics / Television & Home Theater': 9, 'Auto & Home Improvement / Patio & Garden': 82, "Women's Fashion / Clothing": 8, "Men's Clothing, Shoes & Accessories / Shoes": 87, 'Health & Beauty / Fragrance': 26, 'Sports & Outdoors / Outdoors': 15, "Women's Clothing, Shoes & Accessories / Plus Size Clothing": 53, 'Electronics / Video Games': 21, 'For the Home / Patio & Garden': 36, "Men's Fashion / Shoes": 55, "Women's Clothing, Shoes & Accessories / Intimates": 32, 'For the Home / Luggage': 70, 'Auto & Home Improvement / Home Appliances': 80, 'Jewelry & Watches / Jewelry Accessories & Storage': 37, 'Sports & Outdoors / Recreation': 65, 'Grocery, Household & Pets / Food': 10, 'Jewelry & Watches / Gemstone & Pearl Jewelry': 28, "Women's Clothing, Shoes & Accessories / Maternity Clothing": 83, 'For the Home / Art': 66, 'Jewelry & Watches / Diamond Jewelry': 58, 'Entertainment / Books': 50, 'Grocery, Household & Pets / Pets': 2, 'Baby, Kids & Toys / Boys Fashion': 57, 'Health & Beauty / Health Care': 51, 'For the Home / Bedding': 61, 'For the Home / Storage & Organization': 71, 'Collectibles / Coins & Paper Money': 84, 'Jewelry & Watches / Fashion Jewelry': 17, 'Baby, Kids & Toys / Toys': 29, 'For the Home / Home Decor': 39, 'For the Home / Furniture': 68, 'Baby, Kids & Toys / Bedding & Bath': 34, 'Health & Beauty / Hair Care': 33, "Women's Fashion / Intimates": 85, 'Grocery, Household & Pets / Household Essentials': 59, 'Jewelry & Watches / Fine Metal Jewelry': 7, 'Sports & Outdoors / Fan Shop': 43, "Men's Clothing, Shoes & Accessories / Accessories": 64, "Men's Fashion / Clothing": 1, "Women's Clothing, Shoes & Accessories / Shoes": 11, 'Entertainment / Movies & TV': 6, 'Jewelry & Watches / Watches': 41, 'Baby, Kids & Toys / Baby Care': 49, 'Entertainment / Video Games': 52, 'Entertainment / Music': 73, 'Grocery, Household & Pets / Candy & Sweets': 45, "Men's Fashion / Accessories": 4, "Women's Clothing, Shoes & Accessories / Accessories": 23, 'Health & Beauty / Vitamins & Supplements': 30, 'Baby, Kids & Toys / Maternity': 67, 'Electronics / Car Electronics & GPS': 78, 'Electronics / Cell Phones & Accessories': 18, "Men's Clothing, Shoes & Accessories / Clothing": 72, 'Sports & Outdoors / Exercise & Fitness': 19, 'For the Home / Home Appliances': 56, 'Sports & Outdoors / Clothing & Shoes': 76, 'For the Home / Bath': 48, 'Grocery, Household & Pets / Tobacco': 77, 'Entertainment / Magazines': 86, 'Sports & Outdoors / Team Sports': 60, 'Electronics / Office & School Supplies': 22, "Women's Fashion / Plus Size Clothing": 79, 'For the Home / Heating & Cooling': 62, 'For the Home / Furniture & Area Rugs': 74, 'Auto & Home Improvement / Home Improvement': 3, 'Grocery, Household & Pets / Alcohol': 69, 'Sports & Outdoors / Cycling': 38, 'Baby, Kids & Toys / Health & Safety': 75, 'Health & Beauty / Personal Care': 5, 'Baby, Kids & Toys / Girls Fashion': 44, "Women's Clothing, Shoes & Accessories / Clothing": 25, 'Health & Beauty / Massage & Relaxation': 27, 'Grocery, Household & Pets / Beverages': 47, "Women's Fashion / Accessories": 31}

a={'aa':3,'bb':4}
category_dic={}
for key,value in category_dic2.iteritems():
    k=key.split('/')
    sss=str(k[0])+str('>')+str(k[1])
    category_dic[sss]=value

"""
category_dic={'Grocery, Household & Pets': 7, "Women's Fashion": 1, 'For the Home': 9, 'Entertainment': 2, 'Health & Beauty': 6, "Men's Fashion": 8, 'Baby, Kids & Toys': 10, 'Jewelry & Watches': 3, 'Electronics': 0, 'Auto & Home Improvement': 5, 'Sports & Outdoors': 4}

######small category change!!!!!!!!
#category_dic={"Men's Fashion > Accessories": 15, 'Electronics > Portable Audio': 2, "Jewelry & Watches > Men's Jewelry": 64, 'Health & Beauty > Hair Care': 16, 'Grocery, Household & Pets > Household Essentials': 17, 'Health & Beauty > Cosmetics': 41, 'For the Home > Home Decor': 6, 'Entertainment > Video Games': 59, 'Health & Beauty > Sexual Wellness': 36, 'Electronics > Video Games': 48, 'Electronics > Cell Phones & Accessories': 28, 'For the Home > Bedding': 4, 'Sports & Outdoors > Golf': 14, 'Baby, Kids & Toys > Baby Care': 13, 'Health & Beauty > Vitamins & Supplements': 24, 'Health & Beauty > Health Care': 18, "Women's Fashion > Shoes": 23, 'For the Home > Patio & Garden': 30, 'For the Home > Storage & Organization': 0, "Women's Fashion > Plus Size Clothing": 19, 'Electronics > Software': 78, 'Jewelry & Watches > Fashion Jewelry': 62, 'For the Home > Kitchen & Dining': 1, 'For the Home > Bath': 8, "Women's Fashion > Maternity Clothing": 79, "Women's Fashion > Intimates": 12, 'Entertainment > Books': 53, 'Jewelry & Watches > Jewelry Accessories & Storage': 74, 'Entertainment > Magazines': 76, 'Electronics > Musical Instruments': 32, 'Health & Beauty > Massage & Relaxation': 40, 'For the Home > Heating & Cooling': 60, 'Baby, Kids & Toys > Bedding & Bath': 68, "Men's Fashion > Clothing": 5, 'For the Home > Art': 39, 'Sports & Outdoors > Exercise & Fitness': 26, 'Health & Beauty > Skin Care': 31, 'Sports & Outdoors > Cycling': 58, 'Entertainment > Music': 71, 'Grocery, Household & Pets > Beverages': 61, 'Auto & Home Improvement > Home Improvement': 21, "Women's Fashion > Accessories": 7, 'Grocery, Household & Pets > Candy & Sweets': 52, 'Electronics > Television & Home Theater': 49, 'Electronics > Computers & Tablets': 29, 'Health & Beauty > Fragrance': 44, 'Baby, Kids & Toys > Toys': 50, 'Baby, Kids & Toys > Maternity': 56, 'Sports & Outdoors > Team Sports': 73, 'Grocery, Household & Pets > Tobacco': 65, 'Auto & Home Improvement > Automotive': 20, 'Grocery, Household & Pets > Food': 9, 'Baby, Kids & Toys > Boys Fashion': 55, 'Jewelry & Watches > Fine Metal Jewelry': 51, 'For the Home > Home Appliances': 10, 'Jewelry & Watches > Diamond Jewelry': 25, "Men's Fashion > Shoes": 34, 'For the Home > Luggage': 43, "Women's Fashion > Clothing": 38, 'Sports & Outdoors > Fan Shop': 22, 'Collectibles > Coins & Paper Money': 69, 'Grocery, Household & Pets > Pets': 35, 'Sports & Outdoors > Outdoors': 54, 'Jewelry & Watches > Watches': 46, 'Auto & Home Improvement > Patio & Garden': 57, 'Grocery, Household & Pets > Alcohol': 70, 'Health & Beauty > Personal Care': 37, 'Electronics > Camera, Video & Surveillance': 45, 'Electronics > Car Electronics & GPS': 72, 'Entertainment > Movies & TV': 11, 'For the Home > Mattresses & Accessories': 67, 'Electronics > Office & School Supplies': 27, 'For the Home > Furniture': 3, 'Jewelry & Watches > Gemstone & Pearl Jewelry': 47, 'Sports & Outdoors > Recreation': 75, 'Auto & Home Improvement > Home Appliances': 66, 'Health & Beauty > Bath & Body': 42, 'Baby, Kids & Toys > Health & Safety': 77, 'Sports & Outdoors > Clothing & Shoes': 63, 'Baby, Kids & Toys > Girls Fashion': 33}

#category_dic={'Baby, Kids & Toys > Toys': 29, 'Grocery, Household & Pets > Household Essentials': 59, 'For the Home > Home Decor': 39, 'For the Home > Home Appliances': 56, 'Health & Beauty > Health Care': 51, 'Sports & Outdoors > Golf': 20, 'Electronics > Software': 81, 'For the Home > Kitchen & Dining': 63, 'Electronics > Musical Instruments': 54, "Women's Clothing, Shoes & Accessories > Intimates": 32, "Men's Clothing, Shoes & Accessories > Clothing": 72, 'Electronics > Television & Home Theater': 9, "Men's Clothing, Shoes & Accessories > Shoes": 87, 'Grocery, Household & Pets > Tobacco': 77, 'For the Home > Furniture & Area Rugs': 74, "Women's Fashion > Shoes": 24, 'Jewelry & Watches > Watches': 41, 'Electronics > Video Games': 21, 'Auto & Home Improvement > Patio & Garden': 82, 'Electronics > Office & School Supplies': 22, 'For the Home > Furniture': 68, 'For the Home > Heating & Cooling': 62, 'Auto & Home Improvement > Home Appliances': 80, "Men's Clothing, Shoes & Accessories > Accessories": 64, "Men's Fashion > Accessories": 4, 'Electronics > Portable Audio': 14, 'Health & Beauty > Sexual Wellness': 35, 'For the Home > Bedding': 61, 'For the Home > Bath': 48, "Women's Fashion > Intimates": 85, 'Jewelry & Watches > Jewelry Accessories & Storage': 37, 'Baby, Kids & Toys > Bedding & Bath': 34, "Women's Fashion > Plus Size Clothing": 79, "Women's Clothing, Shoes & Accessories > Accessories": 23, "Women's Fashion > Clothing": 8, 'Entertainment > Music': 73, 'Auto & Home Improvement > Automotive': 40, 'Health & Beauty > Bath & Body': 12, 'Jewelry & Watches > Diamond Jewelry': 58, 'Sports & Outdoors > Fan Shop': 43, 'Grocery, Household & Pets > Pets': 2, 'Grocery, Household & Pets > Alcohol': 69, 'Entertainment > Movies & TV': 6, 'For the Home > Storage & Organization': 71, 'Health & Beauty > Skin Care': 42, 'Sports & Outdoors > Recreation': 65, "Jewelry & Watches > Men's Jewelry": 16, 'Health & Beauty > Hair Care': 33, 'Grocery, Household & Pets > Beverages': 47, "Women's Clothing, Shoes & Accessories > Clothing": 25, 'Entertainment > Magazines': 86, "Men's Fashion > Clothing": 1, 'Electronics > Computers & Tablets': 13, 'Health & Beauty > Fragrance': 26, 'Baby, Kids & Toys > Maternity': 67, 'Sports & Outdoors > Team Sports': 60, 'Sports & Outdoors > Cycling': 38, 'Jewelry & Watches > Fine Metal Jewelry': 7, 'Entertainment > Video Games': 52, 'Collectibles > Coins & Paper Money': 84, 'Sports & Outdoors > Outdoors': 15, 'For the Home > Art': 66, 'Health & Beauty > Personal Care': 5, "Women's Clothing, Shoes & Accessories > Maternity Clothing": 83, 'Health & Beauty > Vitamins & Supplements': 30, 'Auto & Home Improvement > Home Improvement': 3, 'Baby, Kids & Toys > Baby Care': 49, 'Baby, Kids & Toys > Health & Safety': 75, 'Health & Beauty > Cosmetics': 0, 'Electronics > Cell Phones & Accessories': 18, 'For the Home > Patio & Garden': 36, 'Jewelry & Watches > Fashion Jewelry': 17, 'Baby, Kids & Toys > Girls Fashion': 44, 'Sports & Outdoors > Exercise & Fitness': 19, "Women's Clothing, Shoes & Accessories > Shoes": 11, "Women's Clothing, Shoes & Accessories > Plus Size Clothing": 53, "Women's Fashion > Accessories": 31, 'Grocery, Household & Pets > Candy & Sweets': 45, 'Jewelry & Watches > Gemstone & Pearl Jewelry': 28, 'Grocery, Household & Pets > Food': 10, 'Baby, Kids & Toys > Boys Fashion': 57, 'For the Home > Luggage': 70, 'Entertainment > Books': 50, 'Electronics > Camera, Video & Surveillance': 46, 'Electronics > Car Electronics & GPS': 78, 'Health & Beauty > Massage & Relaxation': 27, "Men's Fashion > Shoes": 55, 'Sports & Outdoors > Clothing & Shoes': 76}
category_dic2={}
for key,value in category_dic.iteritems():
    category_dic2[value]=key
#print (category_dic2)
lines = open("160819_name_4.txt", "r").read().split('\n')#.decode('utf-8').split('\n')

count=0
count1=0
count0=0
ccq=0
for line in lines:
    sents=line.split('||')
    if (len(sents) <= 2):
            continue
    sent=sents[1]

    #tex = []
    sent = sent.lower()#.encode('ascii', 'ignore')
    sent = "".join(l for l in sent if l not in string.punctuation)
    #sent = sent.translate(None, string.punctuation)
    #sent= str(filter(str.isalpha, sent))
    sent=sent.translate(None, '0123456789')
    sent=sent.replace('’s','')
    sent=str(sent)
    #tex = " ".join([wnl.lemmatize(i) for i in sent.split()])
    #print(sent)
    #sent = ' '.join(wnl.lemmatize(str(word)) for word in sent.split() if wnl.lemmatize(str(word)) not in StopWords)
    sent2=''
    c=0
    for word in sent.split():
        try:
            if wnl.lemmatize(str(word)) not in StopWords:
                sent2=sent2+' '+wnl.lemmatize(str(word))
        except:
            c=c+1

    if c!=0:
       ccq=ccq+1
        #print ("c is",c)
    sent=sent2

    try:
        ww = sents[2].split('>')
        sss2 = (str(ww[0])).strip()
    except:
        #print("count1:",count1)
        count1=count1+1
         #print (sss2)

    try:
        category_dic.get(sss2)
        #a=doc_dic[category_dic.get(sss2)]
        if doc_dic[category_dic.get(sss2)]==[]:
            doc_dic[category_dic.get(sss2)]=sent
        else:
            doc_dic[category_dic.get(sss2)] = str(doc_dic[category_dic.get(sss2)]) + ' ' + sent
    except:
        print(count)
        count=count+1
    #print("count0:",count0)
    count0=count0+1
    #doc_dic[category_dic.get(sss2)]=str(a)+' '+sent

print ("ccq is.....:",ccq)
#print (doc_dic[36])
for i in range(len(doc_dic)):
    if doc_dic[i]==[]:
        continue
    vocab=vocab+doc_dic[i].split()


vocab = list(set(vocab))
print(len(vocab))


def shape(A):
    num_rows=len(A)
    num_cols=len(A[0]) if A else 0

    return num_rows,num_cols

def make_matrix(num_rows,num_cols,entry_fn):
    return [[entry_fn(i,j) for j in range(num_cols)]for i in range(num_rows)]

def is_diagonal(i,j):
    if doc_dic[i] == []:
        return 0
    return doc_dic[i].split().count(vocab[j])

dtm=make_matrix(len(doc_dic),len(vocab),is_diagonal)
print("dtm shape")
print (shape(dtm))
doccount=shape(dtm)[0]
vocabcount=shape(dtm)[1]


global n
n=[]


d=[]
for b in range(doccount):
    sum=0
    sum2=0
    for a in dtm[b]:
        if a>0:
            sum=sum+a
    n.append(sum)

for a in range(vocabcount):
    sum = 0
    for i in range(doccount):
        if dtm[i][a]>0:
            sum=sum+1
   # print(sum)
    d.append(np.log10(float(doccount+1)/float(sum)))


def tf_idf(i):
    tfidf={}
    if n[i]==0:
        return
    for j in range(vocabcount):
        tfidf[vocab[j]] = float(float(dtm[i][j]) / float(n[i]) * d[j])

    new=sorted(tfidf.iteritems(), key=itemgetter(1), reverse=True)
    abc=[]
    #abc1=[]
    abc2=[]
    count=0
    #print(category_dic2.get(i)+'=')
    for j in new:
        #print(j)
        if j[1] > 0:
            if count >= 50:
            # abc1.append(j[0])
                abc2.append(j)
                count = count + 1
            elif count<50:
                abc.append(j[0])
                # abc1.append(j[0])
                abc2.append(j)
                count = count + 1
        if j[1]<0:
            break

    t.write(str(count))
    t.write('||')
    t.write(category_dic2.get(i))
    t.write('||')
    """t.write('=')
    t.write('[')
    for a in abc1:
        t.write("'")
        t.write(str(a))
        t.write("',")
    t.write(']')
    t.write('\n')"""
   # t.write('')
    for a in abc2:
        t.write(str(a[0]))
        t.write(">")
        t.write(str(a[1]))
        t.write('#')

    t.write('\n')
    g.write(category_dic2.get(i))
    g.write('=')
    for a in abc:
        g.write(str(a))
        g.write(",")
    g.write('\n')
    #print (abc)


for i in range(len(doc_dic)):


    tf_idf(i)
