from datetime import datetime, date
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
#--------------data read--------------------#
data = open("car.txt", "r")
cardata = data.read()
cardata = cardata.split("\n")

car_data =[]
for i in cardata:
    j =i.split(",")
    car_data.append(j)

car_data.pop() #eliminate the last element
# print(car_data)

data = open("AirQualityUCI.csv", 'r')
airdata = data.read()

airdata = airdata.replace(',', '.')
airdata = airdata.split()

air_data = []
for i in airdata:
    j = i.split(";")
    j.pop()
    j.pop()
    air_data.append(j)

# print(air_data)
# print(len(air_data))
# print(len(air_data[0]))


#-----------------accuray fucntion-----------#
def accuracy_func(tag, result):
    correct = 0
    incorrect = 0
    for i in range(len(Ycar_data)):
        if (result[i] == Ycar_data[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    accuracy = (correct) / (correct + incorrect)
    return accuracy
    #print("accuracy :" + str(accuracy))

#----------------car data----------------#
# Attribute Values:
#
# buying
# v - high, high, med, low
# maint
# v - high, high, med, low
# doors
# 2, 3, 4, 5 - more
# persons
# 2, 4, more
# lug_boot
# small, med, big
# safety
#low, med, high
#
# Class Values:
#
# unacc, acc, good, vgood


# program object:
# guess the class with attributes


print("=======================Car Data=====================")

#==================preprocessing====================

Xcar_data = []
Ycar_data = []

for i in car_data:
    k = []
    tt =0
    for j in i:
        if(j == 'vhigh'):
            t = 4
        elif(j == 'high'):
            t = 3
        elif(j == 'med'):
            t = 2
        elif(j == 'low'):
            t = 1
        elif(j == '5more'):
            t = 5
        elif(j == 'more'):
            t = 6
        elif(j == 'small'):
            t = 1
        elif(j == 'big'):
            t = 3
        elif(j == 'unacc'):
            tt = 1
        elif(j == 'acc'):
            tt = 2
        elif(j == 'good'):
            tt = 3
        elif(j == 'vgood'):
            tt = 4
        else:
            t = int(j)

        k.append(t)
    k.pop()
    Xcar_data.append(k)
    Ycar_data.append(tt)

# print(Xcar_data)
# print(Ycar_data)

#=====================ZeroR====================
check = [0,0,0,0]
for i in Ycar_data:
    check[i-1] = check[i-1]+1

idx = 0
sum = 0
for i in range(len(check)):
    if(check[i]>check[idx]):
        idx = i
    sum = sum+check[i]

z_acc = check[idx]/sum
print("Accuracy of ZeroR: " + str(z_acc))


#====================OneR======================
check = []
for i in range(len(Xcar_data[0])): # number of attribute
    t = []
    for j in range(7): # value
        tt = []
        for k in range(4): # number of class
            tt.append(0)
        t.append(tt)
    check.append(t)


for i in range(len(Xcar_data)):
    aws = Ycar_data[i]
    for j in range(len(Xcar_data[0])):
        check[j][Xcar_data[i][j]][aws-1]+=1

#print(check)
# [[[0, 0, 0, 0], [258, 89, 46, 39], [268, 115, 23, 26], [324, 108, 0, 0], [360, 72, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [268, 92, 46, 26], [268, 115, 23, 26], [314, 105, 0, 13], [360, 72, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [326, 81, 15, 10], [300, 99, 18, 15], [292, 102, 18, 20], [292, 102, 18, 20], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [576, 0, 0, 0], [0, 0, 0, 0], [312, 198, 36, 30], [0, 0, 0, 0], [322, 186, 33, 35]], [[0, 0, 0, 0], [450, 105, 21, 0], [392, 135, 24, 25], [368, 144, 24, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [576, 0, 0, 0], [357, 180, 39, 0], [277, 204, 30, 65], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]


check_att_acc = [0, 0, 0, 0, 0, 0]

for i in range(len(Xcar_data[0])):

    for j in range(len(check[0])):
        idx = 0
        for k in range(4):
            if(check[i][j][idx] <= check[i][j][k]):
                idx = k
        check_att_acc[i] += check[i][j][idx]

# print(check_att_acc) # [1210, 1210, 1210, 1210, 1210, 1210]

o_acc = check_att_acc[0]/len(Xcar_data)
print("Accuracy of OneR: " + str(o_acc))


#===================Decision Tree=================
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xcar_data, Ycar_data)
result = clf.predict(Xcar_data)

dt_acc = accuracy_func(Ycar_data, result)
print("Accuracy of Decision Tree: " + str(dt_acc))



#==================Naive Bayesian===============
gnb = GaussianNB()
gnb.fit(Xcar_data, Ycar_data)
result = gnb.predict(Xcar_data)

nb_acc = accuracy_func(Ycar_data, result)
print("Accuracy of Naive Bayesian: " + str(nb_acc))


#==================MLP=====================
clf = MLPClassifier(solver= 'lbfgs', alpha= 1e-5, hidden_layer_sizes=(5, 2), random_state= 1)
clf.fit(Xcar_data, Ycar_data)
result = clf.predict(Xcar_data)

mlp_acc = accuracy_func(Ycar_data, result)
print("Accuracy of MLP: " + str(mlp_acc))

#================Logistic Regression=============
reg = linear_model.LogisticRegression()
reg.fit(Xcar_data, Ycar_data)
result = reg.predict(Xcar_data)

LR_acc = accuracy_func(Ycar_data, result)
print("Accuracy of Logistic Regression: " + str(LR_acc))




#----------------Air Quality data----------------#

# Attribute Information:
#
# 0 Date	(DD/MM/YYYY)
# 1 Time	(HH.MM.SS)
# 2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
# 3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
# 4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
# 5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
# 6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
# 7 True hourly averaged NOx concentration in ppb (reference analyzer)
# 8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
# 9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
# 10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
# 11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
# 12 Temperature in Â°C
# 13 Relative Humidity (%)
# 14 AH Absolute Humidity

#program object
#guess the Absolute Humidity with other attributes

print("=======================Air Quality Data=====================")

#====================preprocessing======================

Xair_data = []
Yair_data = []

# cMax = []
# cMin = []
# for i in range(len(air_data[0])-1):
#     cMax.append(-1e10)
#     cMin.append(1e10)

for i in air_data:
    t = []

    date = i[0]
    time = i[1]
    dt = date+' '+time
    d = datetime.strptime(dt, '%d/%m/%Y %H.%M.%S')
    t.append(d.timestamp())

    # if(cMax[0]<t[0]):
    #     cMax[0] = t[0]
    # if(cMin[0]>t[0]):
    #     cMin[0] = t[0]

    for j in range(len(i)-3):
        t.append(float(i[j+2]))

        # if(t[j+1] > -200):
        #     if (cMax[j+1] < t[j+1]):
        #         cMax[j+1] = t[j+1]
        #     if (cMin[j+1] > t[j+1]):
        #         cMin[j+1] = t[j+1]

    Xair_data.append(t)

    k = float(i[14])
    if (k <= -200):
        k = -1
    else:
        k *= 2
        k = int(k)
    Yair_data.append(k)

# print(Xair_data)
print(Yair_data)

# print(cMax)
# print(cMin)
# [1112590800.0, 11.9, 2040.0, 1189.0, 63.7, 2214.0, 1479.0, 2683.0, 340.0, 2775.0, 2523.0, 44.6, 88.7, 2.231]
# [1078909200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0, -200.0]
# [1112590800.0, 11.9, 2040.0, 1189.0, 63.7, 2214.0, 1479.0, 2683.0, 340.0, 2775.0, 2523.0, 44.6, 88.7, 2.231]
# [1078909200.0, 0.1, 647.0, 7.0, 0.1, 383.0, 2.0, 322.0, 2.0, 551.0, 221.0, -1.9, 9.2, 0.1847]

