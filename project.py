import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import csv

ac = [0, 0, 0]
cc = []
f= open("Data_Results.txt","a+")
f.write("--------------------------------------------------------------\n")
dt = pd.read_csv('Ads.csv')
x = dt.iloc[:, [1, 2, 3]].values
y = dt.iloc[:, -1].values
tsize = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=tsize, random_state=10)

def bayes(todo):
    cl='Bayes Net'
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    pr = clf.predict(x_test)

    if(todo==1):
        d1=data_input('Bayes Net')
        ans=clf.predict([[d1[0], d1[1], d1[2]]])
        if(ans==0):
            ans='No'
            print("Predicted Class: No")
        elif(ans==1):
            ans='Yes'
            print("Predicted Class: Yes")
        writeToText(d1[0], d1[1], d1[2], ans, cl)
        return

    count = 0
    match = 0
    for i in range(0, len(pr)):
        if pr[i] == y_test[i]:
            match = match + 1
        count = count + 1
    # print('With Bayes Net')
    # print("Test Size =", tsize)
    # print("Total: ", count)
    # print("Match: ", match)
    # print(f"Accuracy: {match / count * 100}%")
    # print('\n')
    cm = confusion_matrix(y_test, pr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    ac[0] = float(match / count * 100)
    cc.insert(0, 'Bayes Net')
    f.write("\nBayes Net\n")
    f.write("Test Size = %.2f, " % tsize)
    f.write("Total: %d, " % count)
    f.write("Match: %d, " % match)
    f.write("Accuracy: %.2f" % (match / count * 100) + '%\n')


def decision(todo):
    cl='Decision Tree'
    clf = DecisionTreeClassifier(criterion='entropy', random_state=10, )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    if (todo == 1):
        d1 = data_input('Decision Tree')
        ans = clf.predict([[d1[0], d1[1], d1[2]]])
        if (ans == 0):
            ans='No'
            print("Predicted Class: No")
        elif (ans == 1):
            ans='Yes'
            print("Predicted Class: Yes")
        writeToText(d1[0], d1[1], d1[2], ans, cl)
        return

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()


    match = 0
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            match = match + 1
        count = count + 1
    # print('With Decision Tree')
    # print("Test Size =", tsize)
    # print("Total: ", count)
    # print("Match: ", match)
    # print(f"Accuracy: {match / count * 100}%")
    # print('\n')
    ac[1] = float(match / count * 100)
    cc.insert(1, 'Decision Tree')
    f.write("\nDecision Tree\n")
    f.write("Test Size = %.2f, " % tsize)
    f.write("Total: %d, " % count)
    f.write("Match: %d, " % match)
    f.write("Accuracy: %.2f" % (match / count * 100)+'%\n')


def svm(todo):
    cl='Support Vector Machine'
    clf = SVC()
    clf.fit(x_train, y_train)
    pr = clf.predict(x_test)
    if (todo == 1):
        d1 = data_input('Support Vector Machine')
        ans = clf.predict([[d1[0], d1[1], d1[2]]])
        if (ans == 0):
            ans='No'
            print("Predicted Class: No")
        elif (ans == 1):
            ans='Yes'
            print("Predicted Class: Yes")
        writeToText(d1[0], d1[1], d1[2],ans,cl)
        return

    count = 0
    match = 0
    for i in range(0, len(pr)):
        if pr[i] == y_test[i]:
            match = match + 1
        count = count + 1
    # print('With Support Vector Machine')
    # print("Test Size =", tsize)
    # print("Total: ", count)
    # print("Match: ", match)
    # print(f"Accuracy: {match / count * 100}%")
    # print('\n')
    cm = confusion_matrix(y_test, pr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    ac[2] = float(match / count * 100)
    cc.insert(2, 'Support Vector Machine')
    f.write("\nSupport Vector Machine\n")
    f.write("Test Size = %.2f, " % tsize)
    f.write("Total: %d, " % count)
    f.write("Match: %d, " % match)
    f.write("Accuracy: %.2f" % (match / count * 100)+'%\n')


def data_input(cl):
    print('Please Enter Feature Values to Compare with '+cl)
    while(1):
        d1 = input("Gender (Male/Female): ")
        if(d1.lower()=='male'):
            d1=1
        elif(d1.lower()=='female'):
            d1=0
        else:
            print('Invalid Input. Please Try Again.')
            continue
        d2 = input("Age (in Number): ")
        d2=int(d2)
        d3 = input("Estimated Salary (in Number): ")
        d3=int(d3)
        return d1, d2, d3
        break

def compare_all():
    f.write('Comapring All Classifiers...\n')
    bayes(0)
    decision(0)
    svm(0)
    tempo = 0
    ind = 0
    for i in range(0, len(ac)):
        print(cc[i], ':', ac[i], '%')
        if (ac[i] > tempo):
            tempo = ac[i]
            ind = i
    if (ind == 0):
        ind = 'Bayes Net'
    elif (ind == 1):
        ind = 'Decision Tree'
    elif (ind == 2):
        ind = 'Support Vector Machine'
    print()
    print(ind, 'classifier is the winner with accuracy', tempo,'%')
    f.write('\n')
    f.write(ind + ' classifier is the winner with accuracy ' + str(tempo) + '%' + '\n')

def writeToText(d1,d2,d3,ans,cl):
    if(d1==1):
        d1='Male'
    else:
        d1='Female'
    d2=str(d2)
    d3=str(d3)
    f.write('\nClassifier Used: ' +cl +'\n')
    f.write('Data Used--> ')
    f.write('Gender: '+d1+', Age: '+str(d2)+ ', Estimated Salary: '+str(d3)+'\n')
    f.write('Predicted Class: '+ ans)
    f.write('\n')

print('1. Predict With Bayes Net\n2. Predict With Decision Tree\n3. Predict With Support Vector Machine\n4. Test All\n5. Exit')

while (1):
    x = input("\nPlease Enter a Number (1 to 5): ")
    if (x == '1'):
        bayes(1)
    elif (x == '2'):
        decision(1)
    elif (x == '3'):
        svm(1)
    elif (x == '4'):
        compare_all()
    elif (x == '5'):
        print('Thanks for using me.')
        f.close()
        break
    else:
        print('Wrong Input. Please Give a input between 1 to 5!.')

exit()
