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

# i = 0
# while True:
#     i += 1
#     if i > 5:break
#     print('*' *i)

# test_list = ['one', 'two', 'three']
# for i in test_list:
#     print(i)

# a = [(1,2), (3,4), (5,6)]
# for (first, last) in a:
#     print(first + last)

# marks = [90, 25, 67, 45, 80]

# number = 0
# for i in marks:
#     number = number + 1
#     if i < 60:
#         continue
#     print("%d학생은 합격입니다" %number)

# add  = 0
# for i in range(1, 11):
#     add = add + i
# print(add)

# marks = [90, 25, 67, 45, 80]
# for number in range(len(marks)):
#     if marks[number] < 60:
#         continue
#     print("%d축" %(number + 1))
    
# add = 0
# for i in range(1, 101):
#     add = add + i
# print(add)    

# for i in range(2, 10):
#     for j in range(1, 10):
#         print(i*j, end="")

# a = [1,2,3,4]
# result = []
# for num in a:
#     result.append(num*3)
# print(result)    

# a = 0
# for i in range(100):
#     a = a + i
#     if a >= 10:
#         break
# print(a)

# a = [1,2,3,4]
# result = [num*3 for num in a]
# print(result)

# a = [1,2,3,4]
# result = [num*3 for num in a if num%2 == 0]
# print(result)
# [6, 12]

# i = 0
# while True:
#     i += 1
#     if i > 5: break
#     print( i*'*')

# for i in range(101):
#     print(i)
# A = [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
# total = 0
# for score in A:
#     total += score
# average = total/len(A)
# print(average)    

# numbers = [1,2,3,4,5]
# result = []
# # for n in numbers:
# #     if n % 2 ==1:
# #         result.append(n*2)
# #         print(result)
# result = [num*2 for num in numbers if num%2 == 1]
# print(result)

