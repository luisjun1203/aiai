import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(3)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


 

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60
 
test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
# train_csv.value_counts('근로기간')
 
train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'
# train_csv.loc[train_csv['대출등급']=='G', '대출등급'] = 
 
test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'


lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])


lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])


lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])



X_data = train_csv.drop(['대출등급'], axis=1)
y_data = train_csv['대출등급']


# print(y_data.shape) # (96294,)

# print(X_data.shape) # (96294, 13)


y_data = y_data.values.reshape(-1, 1)  
# print(y_data.shape) # (96294, 1)

ohe = OneHotEncoder(sparse=False)
y_data_ohe = ohe.fit_transform(y_data)
# print(y_data_ohe.shape) # (96294, 7)

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_data_ohe.shape[1]])  

w = tf.compat.v1.Variable(tf.random.normal([13, y_data_ohe.shape[1]]), name='weight')  
b = tf.compat.v1.Variable(tf.zeros([y_data_ohe.shape[1]]), name='bias') 

hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)  
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3000
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


# ACC :  0.4076474131306208

# ACC :  0.4394250939830104