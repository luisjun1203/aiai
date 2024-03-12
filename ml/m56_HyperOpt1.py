import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# print(hyperopt.__version__)   # 0.2.7

search_space = {'X1' : hp.quniform('X1', -10, 10, 1),
                'X2' : hp.quniform('X2', -15, 15, 1)}
                # hp.quniform(label, low, high, q)  # uniform : 균등분포
                
# hp.uniform(label, low, high, q) : label로 지정된 입력 값 변수 검색 공간을
#                                   최솟값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.uniform(label, low, high) : 최솟값 low에서 최대값 high까지 정규분포 형태의 검색 공간 설정
# hp.randint(label, upper): 0부터 최대값 upper까지 random한 정수값으로 검색 공간 설정
# hp.loguniform(label, low, high) : exp(uniform(low, high))값을 반환하며, 
#                                   반환값의 log변환 된 값은 정규분포 형태를 가지는 검색 공간 설정, 통상 y값에 많이 사용
 
def objecive_func(search_space):
    X1 = search_space['X1']
    X2 = search_space['X2']
    return_value =  X1**2 - 20*X2

    return return_value

trial_val = Trials()
best = fmin(fn= objecive_func,
            space=search_space,
            algo=tpe.suggest,       # 알고리즘, 디폴트
            max_evals=20,           # 서치 횟수
            trials=trial_val,
            rstate=np.random.default_rng(seed=10)       # 난수생성,  직접 찾아봐
            # rstate=333,
) 
# print(best) # {'X': 0.0, 'y': 15.0}
# print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'},
#  {'loss': 200.0, 'status': 'ok'}, {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'},
#  {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'},
#  {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'},
#  {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'},
#  {'loss': -300.0, 'status': 'ok'}, {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'},
#  {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]
# print(trial_val.vals)
# {'X': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0],
#  'y': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}

# [실습] 이렇게 예쁘게  나오게  만들어봐!!
# 판다스 데이터프레임 사용

# |     iter    |   target  |   X1  |   x2  |
# ----------------------------------------------
# |     1       |     9     |  0     | 1      |
# |     2       |     9     |   0    | 2      |

import pandas as pd
# df = pd.DataFrame(trial_val.vals)
# print(df)
############# 영현씨 방식 #########################################
# print('| iter    |   taget   |   X1  |   X2  |')
# print("----------------------------------------") 
# X1_list = trial_val.vals['X1']
# X2_list = trial_val.vals['X2']

# for idx, data in enumerate(trial_val.results):
#     loss = data['loss']
#     print(f'|{idx:^10}|{loss:^10}|{X1_list[idx]:^10}|{X2_list[idx]:^10}|')      # ^10 :  10칸중에서 가운데 정렬

####################### 선생님 방식 ########################################
target = [aaa['loss'] for aaa in trial_val.results]
# print(target)
# [-216.0, -175.0, 129.0, 200.0, 240.0, -55.0, 209.0, -176.0, -11.0, -51.0, 136.0, -51.0, 164.0, 321.0, 49.0, -300.0, 160.0, -124.0, -11.0, 0.0]    

df = pd.DataFrame(  {'target' : target,
                    'X1': trial_val.vals['X1'],
                    'X2' : trial_val.vals['X2'],
                  })    
print(df)





