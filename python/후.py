
# xy_test = test_datagen.flow_from_directory( # 데이터를 실시간으로 증강하고, 필요에 따라 배치를 생성하여 모델 학습에 활용할 수 있도록 도와주는 역할

# ########################################################################################################
# X = []
# y = []

# for i in range(len(xy_train)):
#     batch = xy_train.next()
#     X.append(batch[0])          # 현재 배치의 이미지 데이터
#     y.append(batch[1])          # 현재 배치의 라벨 데이터
# X = np.concatenate(X, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
# y = np.concatenate(y, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
# ###############################################################################################

# augment_size = 100

# print(X_train[0].shape)     # (28, 28)
# # plt.imshow(X_train[0], 'gray')
# # plt.show()

# X_data = train_datagen.flow(            # np.tile(A, repeat_shape) :  A 배열이 repeat_shape 형태로 반복되어 쌓인 형태 -> 즉, 동일한 내용의 형태를 반복해서 붙여준다
#     np.tile(X_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),             # X (100,28,28,1)
#     np.zeros(augment_size),                                                             # y  (100, ) # 0으로만 채워진 array 생성
#     batch_size=50,
#     shuffle=False,
    
#     ).next()        # 다음 데이터가 없으니 .next()를 하면 X_data에 (100, 28, 28, 1)이 들어감
# # print(X_data)
# # print(X_data.shape) # 튜플형태라서 에러, 왜냐면 flow에서 튜플 형태로 반환해준다
# print(X_data[0].shape)  # (100, 28, 28, 1)
# print(X_data[1].shape)  # (100,) 전부다 0
# print(np.unique(X_data[1],return_counts = True))    # (array([0.]), array([100], dtype=int64))

# print(X_data[0][0].shape)      #  (28, 28, 1)



# #################################################################################
# it = datagen.flow(img,               
#                   batch_size=1,
                  
#                   )

# fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10))           # subplot : 여러장을 한번에 볼수있음 (2행 5열)
# # matplotlib 사용하여 이미지 표시할 subplot 생성
# # fig = figures : 데이터가 담기는 프레임
# # ax = axes : 실제 데이터가 그려지는 캔버스 ,모든 plot은 axes 위에서 이루어져야 한다

# for i in range(2):
#      for j in range(5):
#         batch = it.next()                   # 증강된 데이터 생성해서 batch에 저장
#         image = batch[0].astype('uint8')       # (150,150,3)        # batch 첫번째 이미지 나타냄
    
#         ax[i, j].imshow(image)     # 이미지가 ax라는 놈에 들어간다 i번째 서브플롯에 이미지 표시
#         ax[i, j].axis('off')       # 해당 subplot 축을 숨김
# print(np.min(batch), np.max(batch))    
# plt.show()    






















