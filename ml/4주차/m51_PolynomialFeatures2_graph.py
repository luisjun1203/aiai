import numpy as np
from sklearn.preprocessing import PolynomialFeatures       
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family'] = 'Malgun Gothic'
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# 1. 데이터

np.random.seed(777)
X = 2 * np.random.rand(100, 1) -1       # -1 ~ 1  100개 난수 생성
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1)  # 랜덤값 추가 : 노이즈 추가 y = (3x^2 + 2x + 1 + 노이즈)

# print(X)
# print(y)
pf = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pf.fit_transform(X)
# print(X_poly)

# 2. 모델 구성
# model = LinearRegression()
model2 = LinearRegression()

model = RandomForestRegressor()         # RandomForest 적용하니까 데이터 변형을 안시켜줘도 나름 굿!
# model2 = RandomForestRegressor()

# 3. 훈련

model.fit(X, y)
model2.fit(X_poly, y)

# 원래 데이터 그리기

plt.scatter(X, y, color = 'blue', label  = 'Original Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Example')

# 다항식 회귀 그래프 그리기

X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
X_plot_poly = pf.transform(X_plot)
y_plot = model.predict(X_plot)
y_plot2 = model2.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color = 'red', label = '그냥')
plt.plot(X_plot, y_plot2, color = 'black', label = 'Polynomial Regression')

plt.legend()
plt.show()






# 4. 평가
# result = model.evaluate(X,y)
# result2 = model2.evaluate(X_poly, y)

# print(result)
# print(result2)






