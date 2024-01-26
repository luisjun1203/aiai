# money = 2000
# card = True
# if money >= 3000 or card:
#     print("택시")
# else:
#     print("걸어가라")

# pocket = ['paper', 'cellphone']
# card = True
# if 'money' in pocket:
#     print("taxi")
# else:
#     if card:
#         print("taxi")
#     else:       
#         print("walk")

# treeHit = 0
# while treeHit <10:
#     treeHit = treeHit +1 
#     print("나무를 %d번 찍었습니다" % treeHit)
#     if treeHit ==10:
#         print("나무 넘어갑니다")

# coffee = 10
# money = 300
# while money:
#     print("돈 받았으니 커피를 줍니다")
#     coffee = coffee-1
#     print("남은 커피양은 %d 입니다" % coffee)
#     if coffee ==0:
#         print("커피가 다떨어졌으니 판매를 중단합니다")
#         break

# coffee = 10
# while True:
#     money = int(input("돈 넣어:"))
#     if money ==300:
#         print("커피 줌")
#         coffee = coffee - 1
#     elif money >300:
#         print("거스름돈 %d를 주고 커피를 줍니다" %(money -300))
#         coffee = coffee - 1
#     else:
#         print("돈을 돌려주고 커피 안줌")
#         print("남은 커피 양은 %d 입니다" % coffee)
#         if coffee == 0:
#             print("노커피 노판매")
#             break


# a = 0
# while a < 10:
#     a = a + 1
#     if a % 3 == 0: continue
#     print(a)
    
# while True:
#     print("ctrl+c를 눌러야 while문을 빠져나갈수있습니다")    

# result = 0
# i = 1
# while i <= 1000:
#     if i % 3 ==0: 
#         result += i
#     i += 1
        
# print(result)

i = 0
while True:
    i += 1
    if i > 5:break
    print('*' *i)








