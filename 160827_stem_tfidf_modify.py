import numpy as np
import random
import sys, math
from datetime import datetime
import pickle
from operator import itemgetter
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

class Classifier:
    def __init__(self, featureGenerator):
        self.featureGenerator = featureGenerator
        self._C_SIZE = 0
        self._V_SIZE = 0
        self._classes_list = []
        self._classes_dict = {}
        self._vocab = {}
        #self._vocab_doc = {} # add by Hana
        #self._vocab_after = {}  # add by Hana

        # add by Eunsoo
        #self.re_class_dic = {}
        self.myweight = []
        #self.myweight_dic = {}
        self.re_vocab = {}

    def setClasses(self, trainingData, level):
        for (label, line) in trainingData:
            if label not in self._classes_dict.keys():
                #self._classes_dict[label] = len(self._classes_list)
                self._classes_list.append(label)

        cat_dict_list = open('cat_dict_' + str(level) + '.txt', 'r')
        self._classes_dict = pickle.load(cat_dict_list) # get cat_dict from cat_dict.txt file
        print self._classes_dict

        for i in range(self._classes_dict.__len__()):
            self._classes_list.append(0)
        for key, value in self._classes_dict.iteritems():
            self._classes_list[value] = key

        self._C_SIZE = len(self._classes_list)

        # set myweight (tf-idf weight matrix)
        #self.myweight = np.zeros((, 11))

        # tf-idf word parsing
        tfidf_file = open("tfidf_" + str(level) + ".txt", "r").read().split('\n')

        for a in range(self._classes_dict.__len__()):
            self.myweight.append({})

        i = 0
        stemmer = SnowballStemmer('english')  # add by Hana
        for line in tfidf_file:
            datas = line.split('||')
            try:
                words = datas[2].split('#')
            except:
                break
            for word in words:
                word_weight = word.split('>')
                try:
                    stemmed_word = stemmer.stem(word_weight[0])
                    cat = self._classes_list[datas[1]]
                    self.myweight[cat][stemmed_word] = float(word_weight[1])
                except:
                    break
            i = i + 1

        print("i is done!!!!!!! : ", i)
        #print self.myweight_dic

        return

    def getClasses(self):
        return self._classes_list

    #----------------------------------------------------------------------------------------------------
    def setVocab(self, trainingData):   # set feature & make model
        print('> vocab setting start ...')
        index = 0
        clean_item = '' # add by Hana
        stemmed_item = ''

        for (label, line) in trainingData:
            line = self.featureGenerator.getFeatures(line) # change A(upper) to a(lower), split by ' '
            #line_temp = list(set(line)) # remove duplication from line

            # add by Hana
            for item in line:
                if (item not in self._vocab.keys()):
                    self._vocab[item] = index
                    #self._vocab_doc[item] = 1  # add item to _vocab_doc(map)
                    index += 1
                #else:
                    #self._vocab_doc[item] += 1

        self._V_SIZE = len(self._vocab)

        # add by Eunsoo
        for key, value in self._vocab.iteritems():
            self.re_vocab[value] = key

        print('> vocab setting end ...')
        return

    def getVocab(self):
        return self._vocab

    # add by Hana
    #def getVocabDoc(self):
    #    return self._vocab_doc

    def train(self, trainingData):
        pass

    def classify(self, testData, params):
        pass

    def getFeatures(self, data):
        return self.featureGenerator.getFeatures(data)

    def save_classifier(classifier, level, ratio, num_of_training_data):
        f = open('./../../model/160602/160819_model_nb_stem_' + str(level) + '_' + str(ratio) + '_' + str(num_of_training_data) + '.pickle', 'wb')
        pickle.dump(classifier, f, -1)
        f.close()

class FeatureGenerator:
    def getFeatures(self, text):
        stemmer = SnowballStemmer('english')  # add by Hana
        text = text.lower().encode('ascii', 'ignore') # 160822, ignore unicode
        texts = text.split()
        feature = []

        stop = stopwords.words('english')  # remove stopwords (1)

        for item in texts:
            #item = str(item).translate(None, string.punctuation)  # remove '.', ',', ';' ...
            clean_item = str(filter(str.isalpha, item)) # extract only alpha
            if (clean_item != ""): # remove empty item
                if (clean_item not in stop):  # remove stopwords (2)
                    stemmed_item = stemmer.stem(clean_item)
                    feature.append(stemmed_item)
        return feature


class NaiveBayesClassifier(Classifier):
    def __init__(self, fg, alpha=0.05): # alpha 0.05 (default)
        Classifier.__init__(self, fg)
        self.__classParams = []
        self.__params = [[]]
        self.__alpha = alpha

    def getParameters(self):
        return (self.__classParams, self.__params)

    def setFeature(self, trainingData, level):
        print('> feature setting start ...')
        self.setClasses(trainingData, level)
        self.setVocab(trainingData)
        self.initParameters()
        #self._vocab_after = self._vocab_doc
        print('> feature setting end ...')

    def train(self, trainingData):
        '''
        self.setClasses(trainingData)
        self.setVocab(trainingData)
        self.initParameters()
        '''

        print('> training start ...')

        for (cat, document) in trainingData:
            for feature in self.getFeatures(document):
                self.countFeature(feature, self._classes_dict[cat])

        for i in range(self._C_SIZE):
            for j in range(self._V_SIZE):
                # print self._counts_in_class[i][j]
                # print self._counts_in_class[i][j] = float(self._counts_in_class[i][j]) * float(1.0 + a)
                tfidf_weight = 0.0

                if self._counts_in_class[i][j] <= 0.0:
                    continue

                try:
                    tfidf_weight = self.myweight[i][str(self.re_vocab[j])]
                    print tfidf_weight
                    print "YesYes", self.re_vocab[j], self._classes_list[i]
                except:
                    print "NO", self.re_vocab[j], self._classes_list[i]

                if tfidf_weight > 0:
                    temp = self._counts_in_class[i][j]
                    print type(temp)
                    print type(tfidf_weight)
                    self._counts_in_class[i][j] = float(temp) * (1.0 + float(tfidf_weight)) # add weight

        print('> training end ...')

    def countFeature(self, feature, class_index):
        counts = 1
        self._counts_in_class[class_index][self._vocab[feature]] += counts
        self._total_counts[class_index] += counts
        self._norm += counts

    def classify(self, testData):
        post_prob = self.getPosteriorProbabilities(testData)
        return self._classes_list[self.getMaxIndex(post_prob)]

    def getPosteriorProbabilities(self, testData):
        post_prob = np.zeros(self._C_SIZE)
        for i in range(0, self._C_SIZE):
            for feature in self.getFeatures(testData):
                post_prob[i] += self.getLogProbability(feature, i)
            post_prob[i] += self.getClassLogProbability(i)
        return post_prob

    def getFeatures(self, testData):
        return self.featureGenerator.getFeatures(testData)

    def initParameters(self):
        self._total_counts = np.zeros(self._C_SIZE)
        self._counts_in_class = np.zeros((self._C_SIZE, self._V_SIZE))
        self._norm = 0.0

    def getLogProbability(self, feature, class_index):
        return math.log(self.smooth(self.getCount(feature, class_index), self._total_counts[class_index]))

    def getCount(self, feature, class_index):
        if feature not in self._vocab.keys():
            return 0
        else:
            return self._counts_in_class[class_index][self._vocab[feature]]

    def smooth(self, numerator, denominator):
        return (numerator + self.__alpha) / (denominator + (self.__alpha * len(self._vocab)))

    def getClassLogProbability(self, class_index):
        # add by Eunsoo
        if self._total_counts[class_index] == 0: # add by Eunsoo
            return -1000.0
        else:
            print math.log(self._total_counts[class_index] / self._norm)
            return math.log(self._total_counts[class_index] / self._norm)

    def getMaxIndex(self, posteriorProbabilities):
        maxi = 0
        maxProb = posteriorProbabilities[maxi]
        for i in range(0, self._C_SIZE):
            if (posteriorProbabilities[i] >= maxProb):
                maxProb = posteriorProbabilities[i]
                maxi = i
        return maxi


class Dataset:
    def __init__(self, filename, level, num_of_training_data):
        lines = open(filename, "r").read().decode('utf-8').split('\n')  # add decode('utf-8') 160822 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        num = 0 # total number of data
        wrong = 0
        nutshell = ""
        category = ""
        self.__dataset = []

        for line in lines:
            if num == num_of_training_data: break # number of training data
            else:
                line = lines[num]
                id_nts_ctg = line.split('||') # id || nutshell || category

                # nutshell
                if(id_nts_ctg.__len__() <= 2):
                    #print('> item number is wrong\n')
                    break
                #print(id_nts_ctg[0] + '\n' + id_nts_ctg[1] + '\n' + id_nts_ctg[2] + '\n')
                nutshell = id_nts_ctg[1] # nutshell = '~~~~~~'

                #category
                categories = id_nts_ctg[2].split(" > ") # category = 1 > 2 > 3
                if(categories.__len__() <= 2):
                    #print('> category number is wrong\n')
                    break

                if(level == 1):
                    category = categories[0] # Level 1
                elif(level == 2):
                    category = categories[0] + ' / ' + categories[1] # Level 2
                else:
                    category = categories[0] + ' / ' + categories[1] + ' / ' + categories[2].strip()  # Level 3
                    print categories[2]
                    print categories[2].__len__()

                if(wrong != 1) :
                    self.__dataset.append([category, nutshell])

                num = num + 1
                wrong = 0

        random.shuffle(self.__dataset)
        self.__D_SIZE = num
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

    infile = "./../../../data/160819_name.txt"
    outfile = open("test.txt", 'w')

    if len(sys.argv) > 1:
        infile = sys.argv[1]

    data = Dataset(infile, level, num_of_training_data)

    ratio_training = int(ratio[0] + '0')
    ratio_test = int(ratio[1] + '0')
    data.setTrainSize(ratio_training)
    data.setTestSize(ratio_test)

    train_set = data.getTrainingData()
    test_set = data.getTestData()

    test_data = [test_set[i][1] for i in range(len(test_set))]
    actual_labels = [test_set[i][0] for i in range(len(test_set))]

    fg = FeatureGenerator()
    alpha = 0.5  # smoothing parameter

    nbClassifier = NaiveBayesClassifier(fg, alpha)

    # training start
    now1 = datetime.now()

    nbClassifier.setFeature(train_set, level)
    nbClassifier.train(train_set)

    #nbClassifier._vocab.pop('i')
    #nbClassifier._vocab.pop('is')
    #nbClassifier._vocab.pop('the')

    now2 = datetime.now()
    # training end

    training_time = now2 - now1

    nbClassifier.save_classifier(level, ratio, num_of_training_data)

    # accuracy test
    print('\n> test start ...\n')
    outfile.write('\n> test start ...\n')
    correct = 0
    total = 0

    cat_dict = nbClassifier._classes_dict
    num = cat_dict.__len__()
    eunsoo = np.zeros((num, num))

    for line in test_data:
        line = line.lower().encode('ascii', 'ignore')

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
            outfile.write('\n' + str(total) + '. ' + str(line) + '\n\t' + str(best_label) + ' =/= ' + str(actual_labels[total]))

            # check eunsoo
            #print actual_labels[total]
            #print type(str(actual_labels[total]))
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
    print(eunsoo)
    outfile.write('\n' + str(eunsoo))

    outfile.close()



