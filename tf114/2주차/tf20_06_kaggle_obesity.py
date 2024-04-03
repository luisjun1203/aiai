import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



test_csv.loc[test_csv['CALC']=='Always', 'CALC'] = 'Frequently'

##################### 교통수단 컬럼 살짝 변경 #######################################
train_csv.loc[train_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
train_csv.loc[train_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

test_csv.loc[test_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
test_csv.loc[test_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'


# Bike를 대중교통에 포함시켰다가 Walking으로 바꿈
# print(np.unique(train_csv['MTRANS'], return_counts=True))
# print(np.unique(test_csv['MTRANS'], return_counts=True))


################# 운동량 컬럼 추가 ###################################################################

train_csv['Exercise_Score'] = train_csv['FAF'] - train_csv['TUE'] + train_csv['FCVC']
test_csv['Exercise_Score'] = test_csv['FAF'] - test_csv['TUE'] + test_csv['FCVC']

# print(train_csv['Exercise_Score'])
#################### 식습관 가족력 컬럼 추가 ##############################################

def classify_diet(caec, calc, favc, family_history):
    if family_history == 'yes':
        return 'Moderate'
    elif caec == 'Always' and calc == 'Frequently' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Frequently' and calc == 'Always' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Sometimes' and calc == 'Frequently'and favc == 'yes':
        return 'Moderate'
    elif caec == 'Sometimes' and calc == 'Always'and favc == 'yes':
        return 'Moderate'
    else:
        return 'Healthy'
    
train_csv['Diet_Class'] = train_csv.apply(lambda row: classify_diet(row['CAEC'], row['CALC'], row['FAVC'], row['family_history_with_overweight']), axis=1)  
test_csv['Diet_Class'] = test_csv.apply(lambda row: classify_diet(row['CAEC'], row['CALC'], row['FAVC'], row['family_history_with_overweight']), axis=1)  



lae = LabelEncoder()
lae.fit(train_csv['Gender'])
train_csv['Gender'] = lae.transform(train_csv['Gender'])
test_csv['Gender'] = lae.transform(test_csv['Gender'])



lae.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = lae.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae.transform(test_csv['family_history_with_overweight'])


lae.fit(train_csv['FAVC'])
train_csv['FAVC'] = lae.transform(train_csv['FAVC'])
test_csv['FAVC'] = lae.transform(test_csv['FAVC'])


lae.fit(train_csv['CAEC'])
train_csv['CAEC'] = lae.transform(train_csv['CAEC'])
test_csv['CAEC'] = lae.transform(test_csv['CAEC'])


lae.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = lae.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae.transform(test_csv['SMOKE'])

lae.fit(train_csv['SCC'])
train_csv['SCC'] = lae.transform(train_csv['SCC'])
test_csv['SCC'] = lae.transform(test_csv['SCC'])

lae.fit(train_csv['CALC'])
train_csv['CALC'] = lae.transform(train_csv['CALC'])
test_csv['CALC'] = lae.transform(test_csv['CALC'])

lae.fit(train_csv['MTRANS'])
train_csv['MTRANS'] = lae.transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae.transform(test_csv['MTRANS'])

lae.fit(train_csv['Diet_Class'])
train_csv['Diet_Class'] = lae.transform(train_csv['Diet_Class'])
test_csv['Diet_Class'] = lae.transform(test_csv['Diet_Class'])

# print(train_csv['MTRANS'])
# # print(train_csv['CALC'])
# print(train_csv['SCC'])
# print(train_csv['CAEC'])
# print(train_csv['SMOKE'])

# BMI 컬럼추가
train_csv['BMI'] = 1.3 * (train_csv['Weight'] / (train_csv['Height']**2.5))
test_csv['BMI'] = 1.3 * (test_csv['Weight'] / (test_csv['Height']**2.5))


# print(train_csv.info())
# print(test_csv.info())



X_data = train_csv.drop(['NObeyesdad'], axis=1)
y_data = train_csv['NObeyesdad']

# print(X_data.shape) # (20758, 19)
# print(y_data.shape) # (20758,)
y_data = y_data.values.reshape(-1, 1)  
# print(y_data.shape) # 

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data)
# print(y_data_ohe.shape) # (20758, 7)


scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 19])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_data_ohe.shape[1]])  

# w = tf.compat.v1.Variable(tf.random.normal([19, y_data_ohe.shape[1]]), name='weight')  
# b = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name='bias') 

# hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)

w1 = tf.compat.v1.Variable(tf.random_normal([19,19]), name= 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name = 'bias1')
layer1 = tf.nn.swish(tf.compat.v1.matmul(X, w1) + b1)

# layer2 = model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19,97]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name = 'bias2')
layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)


# layer3 = model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97,9]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name = 'bias3')
layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)

# layer4 = model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9,21]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name = 'bias4')
layer4 = tf.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

# layer5 = model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.random_normal([21,y_data_ohe.shape[1]]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name = 'bias5')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)




loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)  
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2000

for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={X: X_data_scaled, y: y_data_ohe})
    if step % 50 == 0:
        print(step, 'loss : ', cost_val)

y_predict_ohe = sess.run(hypothesis, feed_dict={X: X_data_scaled})
y_predict = np.argmax(y_predict_ohe, axis=1)
original_y_data = np.argmax(y_data_ohe, axis=1)  

acc = accuracy_score(original_y_data, y_predict)
print("ACC : ", acc)

sess.close()


# ACC :  0.8518161672608151

# ACC :  0.8935831968397726
