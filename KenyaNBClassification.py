#Python Code For Naive Bayes Classification
#Importing the Dataset
import numpy as np
import csv
import pandas as pd
import random

random.seed(0)

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

def randList(start, end, num):
    res = [] 
  
    for j in range(num):
        resln = len(res)
        grnum = random.randint(start, end)
        if grnum not in res and resln < 8:
            res.append(grnum)
    reslst = random.sample(res,4)
    return reslst

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def commonWordsInStmtList(statement, words_list):
    stmt_words = statement.split()
    Set1 = set(stmt_words)
    Set2 = set(words_list)      
    exists = False
    common_words = Set2.intersection(Set1)
    #print("common_words: " + str(common_words))
    comwordln = len(common_words)
    #print("comwordln: " + str(comwordln))
    if comwordln == 0:
        exists = False
    elif comwordln > 0:
        exists = True
    return exists, common_words

def keyWordsLists():
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/depression.csv", "r"), delimiter=",")

    xdep = list(reader)
    xdepln = len(xdep)
        
    depression = []
    for row in range(1, xdepln):    
        depression.append(xdep[row][0]) 
                       
    depression = np.array(depression)
        
    #depression = depression[:,0]
    set_depression = set(depression)  
    list_depression = list(set_depression)
    #print("list_depression: " + str(list_depression))
    
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/alcohol.csv", "r"), delimiter=",")

    xalc = list(reader)
    xalcln = len(xalc)
        
    alcohol = []
    for row in range(1, xalcln):    
        alcohol.append(xalc[row][0]) 
                       
    alcohol = np.array(alcohol)
         
    #alcohol = alcohol[:,0]
    set_alcohol = set(alcohol)  
    list_alcohol = list(set_alcohol)   
    #print("list_alcohol: " + str(list_alcohol))

    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/sucide.csv", "r"), delimiter=",")

    xsuc = list(reader)
    xsucln = len(xsuc)
        
    sucide = []
    for row in range(1, xsucln):    
        sucide.append(xsuc[row][0]) 
                       
    sucide = np.array(sucide)
         
    #sucide = sucide[:,0]
    set_sucide = set(sucide)   
    list_sucide = list(set_sucide)
    #print("list_sucide: " + str(list_sucide))
    
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/drugs.csv", "r"), delimiter=",")

    xdrg = list(reader)
    xdrgln = len(xdrg)
        
    drugs = []
    for row in range(1, xdrgln):    
        drugs.append(xdrg[row][0]) 
                       
    drugs = np.array(drugs)
         
    #drugs = drugs[:,0]
    set_drugs = set(drugs)  
    list_drugs = list(set_drugs)
    #print("list_drugs: " + str(list_drugs))
    
    return list_depression, list_alcohol, list_sucide, list_drugs

def keyWordsLists2():
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/depression.csv", "r"), delimiter=",")

    xdep = list(reader)
    xdepln = len(xdep)
        
    depression = []
    for row in range(1, xdepln):    
        depression.append(xdep[row][0]) 
                       
    depression = np.array(depression)
        
    #depression = depression[:,0]
    set_depression = set(depression)  
    list_depression = list(set_depression)
    #print("list_depression: " + str(list_depression))
    
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/alcohol.csv", "r"), delimiter=",")

    xalc = list(reader)
    xalcln = len(xalc)
        
    alcohol = []
    for row in range(1, xalcln):    
        alcohol.append(xalc[row][0]) 
                       
    alcohol = np.array(alcohol)
         
    #alcohol = alcohol[:,0]
    set_alcohol = set(alcohol)  
    list_alcohol = list(set_alcohol)   
    #print("list_alcohol: " + str(list_alcohol))

    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/sucide.csv", "r"), delimiter=",")

    xsuc = list(reader)
    xsucln = len(xsuc)
        
    sucide = []
    for row in range(1, xsucln):    
        sucide.append(xsuc[row][0]) 
                       
    sucide = np.array(sucide)
         
    #sucide = sucide[:,0]
    set_sucide = set(sucide)   
    list_sucide = list(set_sucide)
    #print("list_sucide: " + str(list_sucide))
    
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/drugs.csv", "r"), delimiter=",")

    xdrg = list(reader)
    xdrgln = len(xdrg)
        
    drugs = []
    for row in range(1, xdrgln):    
        drugs.append(xdrg[row][0]) 
                       
    drugs = np.array(drugs)
         
    #drugs = drugs[:,0]
    set_drugs = set(drugs)  
    list_drugs = list(set_drugs)
    #print("list_drugs: " + str(list_drugs))
    
    return list_depression, list_alcohol, list_sucide, list_drugs

def encodeTestData(randlst):
    
    reader = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/Train.csv", "r"), delimiter=",")

    xx = list(reader)
    xxln = len(xx)
        
    allrecs = []
    for row in range(1, xxln):
        fields = []
        recln = len(xx[row])
        for i in range(0, recln):
            fields.append(xx[row][i])    
        allrecs.append(fields)
                       
    #xln = len(x)
    x = np.array(allrecs)
         
    train_ID = x[:,0]
    train_text = x[:,1]
    #txtln = len(train_text)
    #train_label = x[:,2]
    train_lbel = x[:,2]
    set_train_text = set(train_text)
    list_train_text = list(set_train_text)
    lsttxtln = len(list_train_text)
    trainln = len(train_text)
    list_label = ['Depression', 'Alcohol', 'Suicide', 'Drugs']
    print('list_label: ' + str(list_label))
    print('randlst: ' + str(randlst))
    print('                          ')
    print('train-text => sno. randno. label:')
    lbelln = len(train_lbel)
    lstlbelln = len(list_label)
        
    #randlabel = []
    xtrain = []
    x_train_score = []
    for i in range(0, trainln):            
        for j in range(0, lsttxtln):
            if sorted(train_text[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[0]:                                               
                #randlabel.append([randlst[0], list_label[0]])
                x_train_score.append([randlst[0]])
                xtrain.append([train_ID[i], list_train_text[j], list_label[0]])
                print(str(i) + ", " + str(randlst[0]) + ", " + str(list_label[0]))
                break
            elif sorted(train_text[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[1]:
                #randlabel.append([randlst[1], list_label[1]])
                x_train_score.append([randlst[1]])
                xtrain.append([train_ID[i], list_train_text[j], list_label[1]])
                print(str(i) + ", " + str(randlst[1]) + ", " + str(list_label[1]))
                break
            elif sorted(train_text[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[2]:
                #randlabel.append([randlst[2], list_label[2]])
                x_train_score.append([randlst[2]])
                xtrain.append([train_ID[i], list_train_text[j], list_label[2]])
                print(str(i) + ", " + str(randlst[2]) + ", " + str(list_label[2]))
                break
            elif sorted(train_text[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[3]:
                #randlabel.append([randlst[3], list_label[3]])
                x_train_score.append([randlst[3]])
                xtrain.append([train_ID[i], list_train_text[j], list_label[3]])
                print(str(i) + ", " + str(randlst[3]) + ", " + str(list_label[3]))
                break
                
    x_train_score = np.vstack(x_train_score)
    xtrainln = len(x_train_score)    
    y_train_score = []
    print('                 ')
    print('train-text and label nos:')
    for i in range(0, lbelln):            
        for j in range(0, lstlbelln):
            if sorted(train_lbel[i]) == sorted(list_label[j]):
                y_train_score.append([j])
                print(str(i) + ', ' + str(j))
    y_train_score = np.vstack(y_train_score)
    ytrainln = len(y_train_score)
    
    # load the dataset from the CSV file
    reader_test = csv.reader(open("C:/Users/CONALDES/Documents/BasicRightsKenya/Test.csv", "r"), delimiter=",")

    xx_test = list(reader_test)
    xxln_test = len(xx_test)
        
    allrecs_test = []
    for row in range(1, xxln_test):
        fields = []
        recln = len(xx_test[row])
        for i in range(0, recln):
            fields.append(xx_test[row][i])    
        allrecs_test.append(fields)
                       
    #xln = len(x)
    x_test = np.array(allrecs_test)
         
    ID_test = x_test[:,0]
    test_ID = np.vstack(ID_test)
    text_test = x_test[:,1]    
    testln = len(text_test)
    one_hot_labels_test = np.zeros((testln, 4))
    rowl1val = one_hot_labels_test[0]

    print('                 ')
    print('x_test and labels:')
    xtest = []
    temptlist = []
    x_test_score = []
    test_text_labels = np.zeros((testln, 3))
    list_depression, list_alcohol, list_sucide, list_drugs = keyWordsLists()
    list_depression2, list_alcohol2, list_sucide2, list_drugs2 = keyWordsLists2()
    for i in range(0, testln):
        match_seen = False
        for j in range(0, lsttxtln):
            if sorted(text_test[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[0]:
                temptlist.append([randlst[0], list_label[0]])
                x_test_score.append([randlst[0]])
                xtest.append([ID_test[i], text_test[i], list_label[0]])
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[0]))
                match_seen = True
                break
            elif sorted(text_test[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[1]:
                temptlist.append([randlst[1], list_label[1]])
                x_test_score.append([randlst[1]])
                xtest.append([ID_test[i], text_test[i], list_label[1]])
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[1]))
                match_seen = True
                break
            elif sorted(text_test[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[2]:
                temptlist.append([randlst[2], list_label[2]])
                x_test_score.append([randlst[2]])
                xtest.append([ID_test[i], text_test[i], list_label[2]])
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[2]))
                match_seen = True
                break
            elif sorted(text_test[i]) == sorted(list_train_text[j]) and train_lbel[i] == list_label[3]:
                temptlist.append([randlst[3], list_label[3]])
                x_test_score.append([randlst[3]])
                xtest.append([ID_test[i], text_test[i], list_label[3]])
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[3]))
                match_seen = True
                break
                
            
        if match_seen == False:
            dep_exists, dep_words = commonWordsInStmtList(text_test[i], list_depression)
            alc_exists, alc_words = commonWordsInStmtList(text_test[i], list_alcohol)
            suc_exists, suc_words = commonWordsInStmtList(text_test[i], list_sucide)
            drg_exists, drg_words = commonWordsInStmtList(text_test[i], list_drugs)
            if dep_exists == True and 'Depression' == list_label[0]:                   
                temptlist.append([randlst[0], list_label[0]])
                x_test_score.append([randlst[0]])
                xtest.append([ID_test[i], text_test[i], list_label[0]])
                #print('Depression: ' + str(i) + ', ' + str(dep_words))
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[0]))                
            elif alc_exists == True and 'Alcohol' == list_label[1]:
                temptlist.append([randlst[1], list_label[1]])
                x_test_score.append([randlst[1]])
                xtest.append([ID_test[i], text_test[i], list_label[1]])
                #print('Alcohol: ' + str(i) + ', ' + str(alc_words))
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[1]))                
            elif suc_exists == True and 'Sucide' == list_label[2]:
                temptlist.append([randlst[2], list_label[2]])
                x_test_score.append([randlst[2]])
                xtest.append([ID_test[i], text_test[i], list_label[2]])
                #print('Sucide: ' + str(i) + ', ' + str(suc_words))
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[2]))                
            elif drg_exists == True and 'Drugs' == list_label[3]:
                temptlist.append([randlst[3], list_label[3]])
                x_test_score.append([randlst[3]])
                xtest.append([ID_test[i], text_test[i], list_label[3]])
                #print('Drugs: ' + str(i) + ', ' + str(drg_words))
                print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[3]))                
            else:
                dep_exists2, dep_words2 = commonWordsInStmtList(text_test[i], list_depression2)
                alc_exists2, alc_words2 = commonWordsInStmtList(text_test[i], list_alcohol2)
                suc_exists2, suc_words2 = commonWordsInStmtList(text_test[i], list_sucide2)
                drg_exists2, drg_words2 = commonWordsInStmtList(text_test[i], list_drugs2)
                if dep_exists2 == True and 'Depression' == list_label[0]:                   
                    temptlist.append([randlst[0], list_label[0]])
                    x_test_score.append([randlst[0]])
                    xtest.append([ID_test[i], text_test[i], list_label[0]])
                    #print('Depression: ' + str(i) + ', ' + str(dep_words2))
                    print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[0]))
                elif alc_exists2 == True and 'Alcohol' == list_label[1]:
                    temptlist.append([randlst[1], list_label[1]])
                    x_test_score.append([randlst[1]])
                    xtest.append([ID_test[i], text_test[i], list_label[1]])
                    #print('Alcohol: ' + str(i) + ', ' + str(alc_words2))
                    print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[1]))
                elif suc_exists2 == True and 'Sucide' == list_label[2]:
                    temptlist.append([randlst[2], list_label[2]])
                    x_test_score.append([randlst[2]])
                    xtest.append([ID_test[i], text_test[i], list_label[2]])
                    #print('Sucide: ' + str(i) + ', ' + str(suc_words2))
                    print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[2]))
                elif drg_exists2 == True and 'Drugs' == list_label[3]:
                    temptlist.append([randlst[3], list_label[3]])
                    x_test_score.append([randlst[3]])
                    xtest.append([ID_test[i], text_test[i], list_label[3]])
                    #print('Drugs: ' + str(i) + ', ' + str(drg_words2))
                    print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', ' + str(list_label[3]))
                else:
                    grnum = random.randint(1, 4)
                    if grnum == 1:                    
                        temptlist.append([randlst[0], list_label[0]])
                        x_test_score.append([randlst[0]])
                        xtest.append([ID_test[i], text_test[i], list_label[0]])
                        print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', (random) Depression')
                    elif grnum == 2:
                        temptlist.append([randlst[1], list_label[1]])
                        x_test_score.append([randlst[1]])
                        xtest.append([ID_test[i], text_test[i], list_label[1]])
                        print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', (random) Alcohol')
                    elif grnum == 3:
                        temptlist.append([randlst[2], list_label[2]])
                        x_test_score.append([randlst[2]])
                        xtest.append([ID_test[i], text_test[i], list_label[2]])
                        print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', (random) Sucide')
                    elif grnum == 4:
                        temptlist.append([randlst[3], list_label[3]])
                        x_test_score.append([randlst[3]])
                        xtest.append([ID_test[i], text_test[i], list_label[3]])
                        print(str(ID_test[i]) + ', ' + str(text_test[i]) + ', (random) Drugs')
                        
    x_test_score = np.vstack(x_test_score)


    # we set a threshold at 80% of the data
    m = len(x_train_score)
    m_train_set = int(m * 0.8)
        
    print("### Traning Set (80%) and Testing Set (20%) ###")
    print("===============================================")
    print("m_train_set: " + str(roundup(m_train_set,0)))
    print("m_test_set: " + str(roundup((m - m_train_set),0)))                   
    print("                              ")

    # we split the train and test set using the threshold
    X_train, X_val = x_train_score[:m_train_set,:], x_train_score[m_train_set:,:]
    Y_train, y_val = y_train_score[:m_train_set,:], y_train_score[m_train_set:,:]

    print("### Normalized Traning Data (x with biases) ###")
    print("===============================================")
    print("X_train: " + str(X_train))
    print("Y_train: " + str(Y_train))                   
    print("                              ")
    print("### Normalized Testing Data (x_test with biases) ###")
    print("====================================================")
    print("X_val: " + str(X_val))
    print("y_val: " + str(y_val))                       
    print("                              ")
    
    xtestln = len(x_test_score)
    print('xtestln', xtestln)
    temptlist = np.vstack(temptlist)
    templstln = len(temptlist)
             
    #test_texts = np.vstack(temptlist)
    xtrain = np.vstack(xtrain)
    xtest = np.vstack(xtest)
    print('                 ')
    print(str(xtrainln) + ", " + str(xtrainln))
    print(str(templstln) + ", " + str(templstln))
    print('train_ID_text_labels: ')
    for i in range(0, xtrainln):
        print(str(xtrain[i]))
    print('                 ')
    print('test_ID_text_labels: ')
    for i in range(0, testln):
        print(str(xtest[i]))
    print('                 ')       
    print('test_text_labels: ')
    for i in range(0, templstln):
        print(str(temptlist[i]))
    
    return xtrain, xtest, test_ID, X_train, Y_train, X_val, y_val, x_test_score

#data = pd.read_csv("C:/Users/CONALDES/Documents/BasicRightsKenya/Train.csv")

randlst = randList(21, 29, 9)

xtrain, xtest, test_ID, X_train, Y_train, X_val, y_val, x_test_score = encodeTestData(randlst)
data = pd.DataFrame(xtrain, columns = ['ID','text','label'])     # a = pandas.DataFrame(df.sum(), columns = ['whatever_name_you_want'])
#data = data[['ID','text','label']]
print('data', data)

#Exploring The Dataset
'''
print("###################################################################################")
print("\nFeatures/Columns : \n", data.columns)
print("\n\nNumber of Features/Columns : ", len(data.columns))
print("\nNumber of Rows : ",len(data))
print("\n\nData Types :\n", data.dtypes)
print("\nContains NaN/Empty cells : ", data.isnull().values.any())
print("\nTotal empty cells by column :\n", data.isnull().sum(), "\n\n")
print("\n\nNumber of Unique Text : ", len(data['text'].unique()))
print("\n\nNumber of Unique Label : ", len(data['label'].unique()))
print("###################################################################################")
#print("\n\nUnique Salaries:\n", data['salary'].unique())
#print("###################################################################################")
'''

'''
The above code block is to help us understand the kind of data we are dealing with such as 
the number of records or samples, number of features, presence of missing values, unique classes etc'
'''

'''
Cleaning The Data

Now let us clean up the data.

We can see that the experiences are given in strings, so lets convert them to integers in a logical 
way by splitting it in to minimum experience (minimum_exp)  and maxim experience (maximum_exp).Also 
we will label encode the categorical features location and salary. We will then delete the original 
experience column, attach the new ones.
'''

'''
print("\ndata['text'] before transformation: \n", data['text'])
print("\ndata['label'] before transformation: \n", data['label'])
#Label encoding text and label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['text'] = le.fit_transform(data['text'])
data['label'] = le.fit_transform(data['label'])
x_data = data['text']
y_data = data['label']

print("\ndata['text'] after transformation: \n", data['text'])
print("\ndata['label'] after transformation: \n", data['label'])
#print("\nx_data.shape : \n", xx_data.shape)
print("\nx_data.shape : \n", x_data.shape)
print("\ny_data.shape : \n", y_data.shape)
# Accessing elements
print("\nx_data[0] : \n", x_data[0])
print("\ny_data[0] : \n", y_data[0])

# Converting test_texts (list) to dataframe format
df_test_texts = pd.DataFrame(test_texts)
print("\ndf_test_texts: \n", df_test_texts)

#test_texts = pd.Series((i[0] for i in test_texts)) #df = pd.Series((i[0] for i in list))

#typeOfxdata = type(x_data)
#print("\ntypeOfxdata : \n", typeOfxdata)

#Deleting the original ID column
data.drop(['ID'], inplace = True, axis = 1)
data = data[['text', 'label']]
'''

'''
Feature Scaling

We will now scale all the numerical features in the dataset except the target variable which is salary(category).
'''

from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#data[['text']] = sc.fit_transform(data[['text']])

#print("\data : \n", data)


#Creating training and validation sets
'''
#Splitting the dataset into  training and validation sets
from sklearn.model_selection import train_test_split
training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)

print("\ntraining_set : \n", training_set)
print("\nvalidation_set : \n", validation_set)

#classifying the predictors and target variables as X and Y
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

print("\nX_train : \n", X_train)
print("\nY_train : \n", Y_train)
print("\nX_val : \n", X_val)
print("\ny_val : \n", y_val)

print("\nX_train.shape : \n", X_train.shape)
print("\nY_train.shape : \n", Y_train.shape)
print("\nX_val.shape : \n", X_val.shape)
print("\ny_val.shape : \n", y_val.shape)

test_texts = df_test_texts.values

scaler = StandardScaler().fit(test_texts)
test_texts = scaler.transform(test_texts)

print("\ntest_texts: \n", test_texts)
print("\ntest_texts.shape : \n", test_texts.shape)
'''

df_x_train = pd.DataFrame(X_train)
scaler = StandardScaler().fit(df_x_train)
X_train = scaler.transform(df_x_train)
Y_train = pd.DataFrame(Y_train)

df_X_val = pd.DataFrame(X_val)
scaler = StandardScaler().fit(df_X_val)
X_val = scaler.transform(df_X_val)
y_val = pd.DataFrame(y_val)

df_x_test = pd.DataFrame(x_test_score)
scaler = StandardScaler().fit(df_x_test)
X_test = scaler.transform(df_x_test)

print("X_train: ", X_train)
print("Y_train: ", Y_train)
print("X_test: ", X_test)

print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)

'''
The above code block will generate predictors and targets which we can fit our model to train and validate.
Measuring the Accuracy

We will use confusion matrix to determine the correct number of predictions. The accuracy is measured as the total number of correct predictions divided by the total number of predictions. We will define a function to calculate and return the accuracy.
'''

#Initialising the Naive Bayes Classifier

'''
We now have all the data ready to be fitted to the Bayesian classifier. In the below code block we will initialize the Naive Bayes Classifier and fit the training data to it.
'''

#Importing the library
from sklearn.naive_bayes import GaussianNB
#Initializing the classifier
classifier = GaussianNB()
#Fitting the training data
classifier.fit(X_train, Y_train.values.ravel())  # model = forest.fit(train_fold, train_y.values.ravel())

#Predicting for the validation set

#The below code will store the predictions returned by the predict method into y_pred
#print("X_val: before predictions", X_val)

y_pred = classifier.predict(X_val)

print("y_pred: ", y_pred)
#Generating the confusion matrix and printing the accuracy

#The below-given code block will generate a confusion matrix from the predictions and the actual values of the validation set for salary.

#Generating the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)

print("__ACCURACY = ", accuracy(cm))

print("\nX_test: before predictions\n", X_test)
y_pred = classifier.predict(X_test)
print("\ntest_texts predicted\n: ", y_pred)
ID_test = np.vstack(test_ID)
testln = len(y_pred)
one_hot_labels_test = np.zeros((testln, 4))
for i in range(0, testln):        
    if y_pred[i] == 0:                    
        one_hot_labels_test[i][0] = 1
        one_hot_labels_test[i][1] = 0
        one_hot_labels_test[i][2] = 0
        one_hot_labels_test[i][3] = 0
    elif y_pred[i] == 1:
        one_hot_labels_test[i][0] = 0
        one_hot_labels_test[i][1] = 1
        one_hot_labels_test[i][2] = 0
        one_hot_labels_test[i][3] = 0
    elif y_pred[i] == 2:
        one_hot_labels_test[i][0] = 0
        one_hot_labels_test[i][1] = 0
        one_hot_labels_test[i][2] = 1
        one_hot_labels_test[i][3] = 0
    elif y_pred[i] == 3:
        one_hot_labels_test[i][0] = 0
        one_hot_labels_test[i][1] = 0
        one_hot_labels_test[i][2] = 0
        one_hot_labels_test[i][3] = 1
        
one_hot_labels_test = one_hot_labels_test.astype("int")
labels_test = np.vstack(y_pred)
test_Ids_labels = np.concatenate((ID_test, one_hot_labels_test), axis=1)
print("\ntest_texts predicted: \n", test_Ids_labels)
with open("C:/Users/CONALDES/Documents/BasicRightsKenya/ConaldesNBSubmission.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Depression', 'Alcohol', 'Suicide', 'Drugs']) 
    for row in test_Ids_labels:    
        l = list(row)    
        writer.writerow(l)
print("                                          ")
print("### C:/Users/CONALDES/Documents/BasicRightsKenya/ConaldesNBSubmission.csv contains results ###")  

#Output :

#__ACCURACY =  0.48351276881154321

#The Naive Bayesian classifier when fitted with the given data gave an accuracy of 48%. 
