# coding: utf-8

import numpy as np
import random
import sys, math
import time
import pickle
from datetime import datetime

class Classifier:
    def __init__(self, featureGenerator):
        self.featureGenerator = featureGenerator
        self._C_SIZE = 0
        self._V_SIZE = 0
        self._classes_list = []
        self._classes_dict = {}
        self._vocab = {}

        a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        self.re_class_dic={}
        self.myweight = []
        self.myweight_dic = {}
        self.re_vocab={}
        a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


    def setClasses(self, trainingData):
        for (label, line) in trainingData:
                if label not in self._classes_dict.keys():
                        self._classes_dict[label] = len(self._classes_list)
                        self._classes_list.append(label)

                self._C_SIZE = len(self._classes_list)

        #self._C_SIZE = len(self._classes_list)"""
        print self._classes_dict
        self._classes_list = []

        print "aaaaaaaaaaaaaaaaaaaaaa"
        #print self._classes_dict
        self._classes_dict={}

        self._classes_dict={'Auto & Home Improvement / Patio & Garden': 75, 'Baby, Kids & Toys / Girls Fashion': 30, 'Baby, Kids & Toys / Bedding & Bath': 48, 'Health & Beauty / Personal Care': 29, 'Electronics / Car Electronics & GPS': 76, 'For the Home / Home Decor': 44, 'Electronics / Cell Phones & Accessories': 35, 'Electronics / Musical Instruments': 15, 'Sports & Outdoors / Outdoors': 36, "Women's Fashion / Plus Size Clothing": 49, 'Electronics / Video Games': 52, 'Grocery, Household & Pets / Candy & Sweets': 43, 'Auto & Home Improvement / Automotive': 12, 'For the Home / Furniture': 6, 'For the Home / Patio & Garden': 37, 'Electronics / Portable Audio': 16, 'Health & Beauty / Hair Care': 7, 'Health & Beauty / Cosmetics': 0, 'Sports & Outdoors / Exercise & Fitness': 53, 'Health & Beauty / Sexual Wellness': 1, "Women's Fashion / Intimates": 13, 'Grocery, Household & Pets / Household Essentials': 24, 'Sports & Outdoors / Clothing & Shoes': 61, 'Health & Beauty / Skin Care': 25, 'Health & Beauty / Massage & Relaxation': 58, 'Entertainment / Video Games': 73, 'For the Home / Luggage': 60, 'For the Home / Bath': 47, 'For the Home / Heating & Cooling': 63, "Women's Fashion / Shoes": 19, 'Grocery, Household & Pets / Tobacco': 55, 'Jewelry & Watches / Fine Metal Jewelry': 56, 'Electronics / Computers & Tablets': 14, 'Entertainment / Magazines': 74, 'Jewelry & Watches / Jewelry Accessories & Storage': 71, 'Grocery, Household & Pets / Alcohol': 67, "Men's Fashion / Clothing": 28, 'Sports & Outdoors / Recreation': 64, "Jewelry & Watches / Men's Jewelry": 62, 'Grocery, Household & Pets / Food': 50, 'Sports & Outdoors / Team Sports': 65, 'Electronics / Office & School Supplies': 2, 'Health & Beauty / Bath & Body': 57, 'Health & Beauty / Fragrance': 27, "Men's Fashion / Shoes": 17, 'Jewelry & Watches / Gemstone & Pearl Jewelry': 69, 'Sports & Outdoors / Golf': 21, 'Jewelry & Watches / Watches': 39, 'Electronics / Camera, Video & Surveillance': 33, "Women's Fashion / Maternity Clothing": 78, 'Baby, Kids & Toys / Maternity': 68, 'Auto & Home Improvement / Home Appliances': 77, 'Baby, Kids & Toys / Boys Fashion': 34, 'For the Home / Home Appliances': 42, 'Baby, Kids & Toys / Baby Care': 46, 'Auto & Home Improvement / Home Improvement': 20, 'For the Home / Art': 10, 'Health & Beauty / Vitamins & Supplements': 26, 'Sports & Outdoors / Fan Shop': 38, 'Jewelry & Watches / Diamond Jewelry': 23, 'Entertainment / Music': 54, 'Electronics / Software': 41, 'Sports & Outdoors / Cycling': 70, 'Entertainment / Books': 72, 'Baby, Kids & Toys / Health & Safety': 59, 'Grocery, Household & Pets / Pets': 9, "Women's Fashion / Accessories": 32, "Women's Fashion / Clothing": 4, "Men's Fashion / Accessories": 31, 'For the Home / Mattresses & Accessories': 66, 'Health & Beauty / Health Care': 11, 'For the Home / Bedding': 40, 'For the Home / Storage & Organization': 45, 'Jewelry & Watches / Fashion Jewelry': 22, 'Grocery, Household & Pets / Beverages': 51, 'Baby, Kids & Toys / Toys': 3, 'Entertainment / Movies & TV': 5, 'For the Home / Kitchen & Dining': 8, 'Electronics / Television & Home Theater': 18}

        for i in range(79):
            self._classes_list.append(0)
        for key,value in self._classes_dict.iteritems():
            self._classes_list[value]=key
        self._C_SIZE = len(self._classes_list)
        print self._classes_dict
        print  self._C_SIZE
        a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        file = open("tfidf3.txt","r").read().split('\n')
        for key, value in self._classes_dict.iteritems():
            self.re_class_dic[value] = key
        print(self.re_class_dic)

        for a in range(len(self._classes_list)):
            self.myweight.append({})
        i = 0
        for category0 in file:
            category=category0.split('||')
            #print category[1]
            try:
                self.myweight_dic[category[1]] = i
            except:
                break
            try:
                abc = category[2].split('#')
            except:
                break
            for k in abc:
                k = k.split('>')
                try:
                    self.myweight[i][k[0]] = k[1]
                except:
                    break

            i = i + 1

        print("i is done!!!!!!!: ", i)
        print self.myweight_dic


        return

    def getClasses(self):
        return self._classes_list

    def setVocab(self, trainingData):
        index = 0;
        for (label, line) in trainingData:
            line = self.featureGenerator.getFeatures(line)
            for item in line:
                if (item not in self._vocab.keys()):
                    self._vocab[item] = index
                    index += 1
        self._V_SIZE = len(self._vocab)
        for key,value in self._vocab.iteritems():
            self.re_vocab[value]=key
        return

    def getVocab(self):
        return self._vocab

    def train(self, trainingData):
        pass

    def classify(self, testData, params):
        pass

    def getFeatures(self, data):
        return self.featureGenerator.getFeatures(data)

    def save_classifier(classifier):
        f = open('model_3.pickle', 'wb')
        pickle.dump(classifier, f, -1)
        f.close()

class FeatureGenerator:
    def getFeatures(self, text):
        text = text.lower()
        text = text.translate(None, '0123456789')
        text = text.replace('’s', '')
        return text.split()


class NaiveBayesClassifier(Classifier):
    def __init__(self, fg, alpha=0.05): # alpha 0.05 (default)
        Classifier.__init__(self, fg)
        self.__classParams = []
        self.__params = [[]]
        self.__alpha = alpha

    def getParameters(self):
        return (self.__classParams, self.__params)

    def train(self, trainingData):
        self.setClasses(trainingData)
        self.setVocab(trainingData)
        self.initParameters()


        for (cat, document) in trainingData:
            for feature in self.getFeatures(document):
                self.countFeature(feature,  self._classes_dict[cat],cat)
        print self._counts_in_class
        for i in range(self._C_SIZE):
            for j in range(self._V_SIZE):
                #print self._counts_in_class[i][j]
                #print self._counts_in_class[i][j] = float(self._counts_in_class[i][j]) * float(1.0 + a)
                a=0
                try:
                    apple=self.myweight[self.myweight_dic[self.re_class_dic[i]]][str(self.re_vocab[j])]
                    print apple
                    print "YesYes", self.re_vocab[j],self.re_class_dic[i]
                except:
                    print "NO",self.re_vocab[j],self.re_class_dic[i]
                if a>0:
                    b=self._counts_in_class[i][j]
                    print type(b)
                    print type(a)
                    self._counts_in_class[i][j] = float(b) * (1.0 + float(apple))

        print self._counts_in_class
    def countFeature(self, feature,  class_index,cat):
        counts = 1
        #print self._C_SIZE
        #print(class_index,"++++++++",self._classes_dict[cat],"++++++++",cat)
        self._counts_in_class[class_index][self._vocab[feature]] = self._counts_in_class[class_index][self._vocab[feature]] + counts


        self._total_counts[class_index] = self._total_counts[class_index] + counts
        self._norm = self._norm + counts

    def classify(self, testData):
        post_prob = self.getPosteriorProbabilities(testData)
        #for a in post_prob:
            #print a
        #print post_prob
        return self._classes_list[self.getMaxIndex(post_prob)]

    def getPosteriorProbabilities(self, testData):
        post_prob = np.zeros(self._C_SIZE)
        for i in range(0, self._C_SIZE):
            for feature in self.getFeatures(testData):
                post_prob[i] += self.getLogProbability(feature, i)
            post_prob[i] += self.getClassLogProbability(i)
        #print post_prob
        return post_prob

    def getFeatures(self, testData):
        return self.featureGenerator.getFeatures(testData)

    def initParameters(self):
        self._total_counts = np.zeros(self._C_SIZE)
        self._counts_in_class = np.zeros((self._C_SIZE, self._V_SIZE))
        self._norm = 0.0


    def getLogProbability(self, feature, class_index):
        """try:
            weight=self.myweight[self.myweight_dic[self.re_class_dic[class_index]]][feature]###########################
        except:
            weight=0.0000001"""
        if self._total_counts[class_index] == 0:
            return 0
        else:
           # print class_index,self.re_class_dic[class_index]
           # print self._total_counts[class_index],"+++++++++",self.getCount(feature,class_index)
            return math.log(self.smooth(self.getCount(feature, class_index), self._total_counts[class_index]))

    def getCount(self, feature, class_index):
        if feature not in self._vocab.keys():
            return 0
        else:
            return self._counts_in_class[class_index][self._vocab[feature]]

    def smooth(self, numerator, denominator):
        return (numerator + self.__alpha)/ (denominator + (self.__alpha * len(self._vocab)))

    def getClassLogProbability(self, class_index):
        #print class_index, self.re_class_dic[class_index]
        #print self._total_counts[class_index]
        if self._total_counts[class_index]<=0.1:
            return -1000.0
        else:
            return math.log(self._total_counts[class_index] / self._norm)

    def getMaxIndex(self, posteriorProbabilities):
        maxi = 0
        maxProb = posteriorProbabilities[maxi]
        for i in range(0, self._C_SIZE):
            if (posteriorProbabilities[i] >= maxProb):
                maxProb = posteriorProbabilities[i]
                maxi = i
        #print maxi,"aa""aa",maxProb
        #print self._classes_list[maxi]
        #print self.re_class_dic[maxi]
        return maxi


class Dataset:
    def __init__(self, filename, level, num_of_training_data):
        lines = open(filename, "r").read().split('\n')
        nutshell = ""
        category = ""
        self.__dataset = []
        before_dataset = []

        # add ny Hana (parsing input data and make before_dataset(not shuffle))
        for line in lines:
            id_nts_ctg = line.split('||')  # id || nutshell || category

            # nutshell
            if (id_nts_ctg.__len__() <= 2): continue  # constraint1

            nutshell = id_nts_ctg[1]  # nutshell = '~~~~~~'

            # add constraints by Hana (160722) : remove sentenses under 2 words
            nutshells = nutshell.split(' ')
            if (nutshells.__len__() <= 2): continue  # constraint2

            # category
            categories = id_nts_ctg[2].split(" > ")  # category = 1 > 2 > 3
            if (categories.__len__() <= 2): continue  # constraint3

            if (level == 1):
                category = categories[0]  # Level 1
            elif (level == 2):
                category = categories[0] + ' / ' + categories[1]  # Level 2
            else:
                category = categories[0] + ' / ' + categories[1] + ' / ' + categories[2]  # Level 3

            before_dataset.append([category, nutshell])

        random.shuffle(before_dataset)
        self.__dataset = before_dataset[0:num_of_training_data]
        self.__D_SIZE = num_of_training_data
        self.__trainSIZE = int(0.6 * self.__D_SIZE)
        self.__testSIZE = int(0.3 * self.__D_SIZE)
        self.__devSIZE = 1 - (self.__trainSIZE + self.__testSIZE)

    def setTrainSize(self, value):
        self.__trainSIZE = int(value * 0.01 * self.__D_SIZE)
        return self.__trainSIZE

    def setTestSize(self, value):
        self.__testSIZE = int(value * 0.01 * self.__D_SIZE)
        return self.__testSIZE

    def setDevelopmentSize(self):
        self.__devSIZE = int(1 - (self.__trainSIZE + self.__testSIZE))
        return self.__devSIZE

    def getDataSize(self):
        return self.__D_SIZE

    def getTrainingData(self):
        return self.__dataset[0:self.__trainSIZE]

    def getTestData(self):
        return self.__dataset[self.__trainSIZE:(self.__trainSIZE + self.__testSIZE)]

    def getDevData(self):
        return self.__dataset[0:self.__devSIZE]


# ============================================================================================

if __name__ == "__main__":

    level = int(raw_input('level (1-3) : '))
    ratio = str(raw_input('ratio of training:test (82, 73, 64, 55) : '))
    num_of_training_data = int(raw_input('number of training data : '))
    print('\n')

    infile = "160819_name_4.txt"
    #outfile = open("./../../output/160429_1.txt", 'w')
    outfile = open('test.txt', 'w')

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    data = Dataset(infile, level, num_of_training_data)

    ratio_training = int(ratio[0] + '0')
    ratio_test = int(ratio[1] + '0')
    data.setTrainSize(ratio_training)
    data.setTestSize(ratio_test)

    train_set = data.getTrainingData()
    test_set = data.getTestData()

#    print(train_set[0:10])
#    print('\n')
#    print(test_set[0:10])
#    print('\n')

    test_data = [test_set[i][1] for i in range(len(test_set))]
    actual_labels = [test_set[i][0] for i in range(len(test_set))]

    fg = FeatureGenerator()
    alpha = 0.5  # smoothing parameter

    nbClassifier = NaiveBayesClassifier(fg, alpha)

    # training start
    now1 = datetime.now()
    nbClassifier.train(train_set)
    #print(nbClassifier._classes_dict)
    now2 = datetime.now()
    # training end

    training_time = now2 - now1

    #nbClassifier.save_classifier(level, ratio, num_of_training_data)

    # accuracy test
    print('\n> test start ...\n')
    outfile.write('\n> test start ...\n')
    correct = 0
    total = 0

    cat_dict = nbClassifier._classes_dict
    num = cat_dict.__len__()
    eunsoo = np.zeros((num, num))

    for line in test_data:
        best_label = nbClassifier.classify(line)
        # print(str(total) + '. ' + line + '\n\t' + best_label + ' =?= ' + actual_labels[total])
        # outfile.write('\n' + str(total) + '. ' + line + '\n\t' + best_label + ' =?= ' + actual_labels[total])
        if best_label == actual_labels[total]:
            correct += 1
            # print('O')
            # outfile.write(' -> O')
        else:
            # print('X')
            # outfile.write(' -> X')

            print(str(total) + '. ' + str(line) + '\n\t' + str(best_label) + ' =/= ' + str(actual_labels[total]))
            outfile.write(
                '\n' + str(total) + '. ' + str(line) + '\n\t' + str(best_label) + ' =/= ' + str(actual_labels[total]))

            # check eunsoo
            # print actual_labels[total]
            # print type(str(actual_labels[total]))
            '''if cat_dict.has_key(actual_labels[total]) != True: # There is no test category in training category(cat_dict)
                cat_dict[actual_labels[total]] = num - 1
                num += 1
                print 'num : ' + str(num)
                for i in range(initial_num): # add row in eunsoo
                    eunsoo[num - 1][i] = 0
                print eunsoo.shape()
'''
            actual_label = cat_dict[actual_labels[total]]
            print 'act index : ' + str(actual_label)
            wrong_label = cat_dict[best_label]
            print 'wrong index : ' + str(wrong_label)
            eunsoo[actual_label][wrong_label] += 1

        total += 1

    acc = 1.0 * correct / total

    print('\n> test end ...\n')
    outfile.write('\n> test end ...\n')
    print('=' * 60)
    outfile.write('=' * 60)
    print(' RESULT')
    outfile.write('\n\n RESULT')
    print('=' * 60)
    outfile.write('\n')
    outfile.write('=' * 60)
    print(' - Level : ' + str(level))
    outfile.write('\n - Level : ' + str(level))
    print(' - Ratio of training:test : ' + str(ratio_training) + ':' + str(ratio_test))
    outfile.write('\n - Ratio of training:test : ' + str(ratio_training) + ':' + str(ratio_test))
    print(' - Amount of data : ' + str(num_of_training_data))
    outfile.write('\n - Amount of data : ' + str(num_of_training_data))
    temp1 = ' - Training time : %d' % training_time.total_seconds()
    print(temp1)
    outfile.write('\n' + temp1)
    temp2 = ' - Accuracy : %0.3f' % acc
    print(temp2)
    outfile.write('\n' + temp2)
    print(' - Amount of Category : ' + str(nbClassifier.getClasses().__len__()))
    outfile.write('\n - Amount of Category : ' + str(nbClassifier.getClasses().__len__()))
    print('=' * 60)
    outfile.write('\n')
    outfile.write('=' * 60)

    print('\n' + str(cat_dict))
    outfile.write('\n' + str(cat_dict))
    print('\n')
    for i in eunsoo:
        print i
   
    outfile.write('\n' + str(eunsoo))

    outfile.close()




