import pandas as pd

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]


index = ["031", "032", "033", "045", "023"]
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
print("==================================================")
# df[0] # key error      
# print(df["031"])  # key error
print(df["종목명"])      #### 판다스의 기준은 열 ####


#### 아모레를 출력하고 싶을때 ####
#print(df[4, 0])    #에러
# print(df["종목명","045"])  #key error
print(df["종목명"]["045"])  
#### 순서는 판다스 열행, 판다스 열행, 판다스 열행 ####

# loc  : 인덱스 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출
    #   intloc                     #int=정수    

print("==================================================")
print(df.loc["031"])
print("=====================아모레 뽑자=============================")
# print(df.loc[3])    #key error
print(df.loc["045"])
print(df.iloc[3])
print("====================네이버 뽑자==============================")
print(df.loc["023"])
print(df.iloc[-1])
print("================아모레 시가(3500) 뽑자==================================")
print(df.loc["045"].loc["시가"])
print(df.loc["045"].iloc[1])
print(df.iloc[3].iloc[1])
print(df.iloc[3].loc["시가"])
print(df.loc["045", '시가'])

print(df.loc["045"][1])     # wanrning뜨지만 가능하긴함
print(df.iloc[3][1])        #  wanrning뜨지만 가능하긴함

print(df.loc["045"]["시가"])
print(df.iloc[3]["시가"])


print(df.loc["045", "시가"])
print(df.iloc[3, 1])

print("================아모레와 네이버의 시가 뽑자==================================")
print(df.iloc[3:5, 1])
print(df.iloc[[3,4], 1])
# print(df.iloc[3:5, "시가"]) # error
# print(df.iloc[[3,4], "시가"])

# print(df.loc[3:5, "시가"])      # error
# print(df.loc[["아모레","네이버"],1])    # error
# print(df.loc[["045":"023"], 1])       #error
print(df.loc["045":"023"], 1)
print(df.loc["045":"023"], "시가")


