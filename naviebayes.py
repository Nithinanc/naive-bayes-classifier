#imports 
import numpy as np
import string
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
from nltk.tokenize import RegexpTokenizer
from nltk.text import Text
from nltk.stem import WordNetLemmatizer
import copy 

#imports end
##train test load
train = np.load("./data/data_train.pkl",allow_pickle=True)
test=np.load("./data/data_test.pkl",allow_pickle=True)
x_train=train[0][:60000]
y_train=train[1][:60000]
x_valid=train[0][60000:]
y_valid=train[1][60000:]
##train test load

##required classes 
stop_words = set(stopwords.words('english')) 
tokenizer = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
SnowballStemmer = SnowballStemmer("english") 
##required class end 
class navieBayes():
    def __init__(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train

    def TrainNavie(self):
        print("Started Traning")
        for index,temp in enumerate(self.x_train):
            self.x_train[index]=self.preProcess(temp)
        self.log_prior=self.calculate_prior(self.y_train)
        self.class_frequencies=self.bag_of_word(self.x_train,self.y_train)
        #self.class_frequencies=self.curdownFreq(self.class_frequencies)
        #print(log_prior)
        #print(len(class_frequencies))
        #print(class_frequencies[0].keys())
        self.class_vocab,self.total_vocab=self.calculate_class_vocab(self.class_frequencies)
        print(self.class_vocab)
        print(self.total_vocab)
        print("Train completed ")
        

    def preProcess(self,content):
        content=content.lower() #to lower case 
        content=re.sub(r'\d+', '', content) #remove digits 
        content=content.translate(str.maketrans('', '', string.punctuation))#remove puctuations 
        content=content.strip()#remove extra space 
        return content

    def Tokenize(self,content):
        tokens = tokenizer.tokenize(content)## remove if nltk is restricted and deelop new method
        tokens = [w for w in tokens if not w in stop_words] #remove stop words
        #tokens = [wordnet_lemmatizer.lemmatize(w) for w in tokens] #lemmitization
        #tokens = [SnowballStemmer.stem(w) for w in tokens]
        NLTKText = Text(tokens)## remove if nltk is restricted develop new method
        return NLTKText.vocab()

    def calculate_prior(self,y_train):
        classes=np.unique(y_train,return_counts=True)
        self.unique_class_Names=classes[0]
        self.class_counts=classes[1]
        log_prior=[]
        for i in range (len(classes[0])):
            #print()
            log_prior.append(np.log(self.class_counts[i]/len(self.unique_class_Names)))
        return log_prior
    
    def curdownFreq(self,class_frequencies):
        filtered_table=[]
        for classtable in class_frequencies:
            dummyclassfreq=copy.deepcopy(classtable)
            for j in classtable.items():
                word=j[0]
                #print(j)
                wordcount=j[1]
                if(wordcount<2):
                    del dummyclassfreq[word]
            filtered_table.append(dummyclassfreq)
        return filtered_table

    def bag_of_word(self,x_train,y_train):
        class_frequencies=[]
        for label in self.unique_class_Names:
            label_list=np.where(np.array(y_train)==label)[0]
            text=""
            for i in label_list:
                #x_train[i]=x_train[i]
                text+=x_train[i]+"\n"
            classwordsfrequencies=(self.Tokenize(text))
            class_frequencies.append(classwordsfrequencies)
        return class_frequencies

    def calculate_class_vocab(self,class_frequencies):
         vocab=set()
         class_vocab=[]
         #cl_fre=[]
         #class_vocab_names=[]
         for rowIndex,data in enumerate(class_frequencies):
             class_vocab.append(sum(data.values()))
            # class_vocab_names.append(set(data.keys()))
             vocab=vocab.union(data.keys())
         #for i in len(class_vocab_names):
         #    cl_fre[i]=vocab-class_vocab_names[i]
         return class_vocab,len(vocab)

    def predict(self,test_data):
         test_data = self.preProcess(test_data)
         fre=self.Tokenize(test_data)
         label_score=[]
         #print(self.class_frequencies[0]['enjoy'])
         for i in range(len(self.unique_class_Names)):
             word_label_score=[]
             class_word_freq=self.class_frequencies[i]
             for j in fre.items():
                 word=j[0]
                 wordcount=j[1]
                 class_word_occurence=0
                 if word in class_word_freq.keys():
                     class_word_occurence=class_word_freq[word]
                 p_i=(class_word_occurence+0.25)/(self.class_vocab[i]+(self.total_vocab*0.25))
                 word_score=wordcount*np.log(p_i)
                 word_label_score.append(word_score)
             label_score.append(sum(word_label_score)+self.log_prior[i])
         return label_score.index(max(label_score))
                 

Test=navieBayes(x_train,y_train)
Test.TrainNavie()
#count=0
#for row,i in enumerate(x_train[100:500]):
 #   test=Test.predict(i)
  #  pred_label=Test.unique_class_Names[test]
   # if(pred_label==y_train[100+row]):
    #    count+=1
#print(count)
def report_predict_test(test,filename="Submission.csv"):
    print("prediction_started")
    csvfile=open(filename,'w', newline='')
    obj=csv.writer(csvfile)
    obj.writerow(("Id","Category"))
    for rowIndex,test_sample in enumerate(test):
        test=Test.predict(test_sample)
        print(rowIndex)
        pred_class=Test.unique_class_Names[test]
        obj.writerow((rowIndex,pred_class))
    csvfile.close()


def validate(x_valid,y_valid):
    accuracy=0
    #print(len(x_valid))
    for rowIndex,test_sample in enumerate(x_valid):
         test=Test.predict(test_sample)
         print(rowIndex)
         pred_class=Test.unique_class_Names[test]
         if(pred_class==y_valid[rowIndex]):
            accuracy+=1
    return accuracy/len(x_valid)

acc=validate(x_valid,y_valid)
print(acc)
#report_predict_test(test,"abc.csv")
#print(len(x_train))
#hyper=range(100)
#hyper/=100
#best_accuracy_h=0
#for i in hyper:
#accuracy=validate(x_valid,y_valid)
#print(accuracy)

#print(accuracy)
#print(x_train[1])
#a=Test.Tokenize(Test.preProcess(x_train[1]))
#print(a.items())
""" y_test=[]
with open('abc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            y_test.append(row[1])
            line_count += 1
    #print(f'Processed {line_count} lines.')
accuracy=validate(test,y_test)
print(accuracy)
 
 """
