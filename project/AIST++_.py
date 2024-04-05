import pickle

# 파일 경로 설정
file_path = 'c:\\_data\\main_project\\20210308_fullset\\aist_plusplus_final\\motions\\gBR_sBM_cAll_d04_mBR0_ch01.pkl'

# 파일 열기 및 내용 불러오기
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 불러온 데이터의 타입과 기본 정보 출력
type(data), len(data) if isinstance(data, (list, dict)) else "N/A"

print(data)


# 딕셔너리의 키들을 리스트로 반환
keys = list(data.keys())

# 딕셔너리에 포함된 키 출력
# print(keys)


data_info = {key: {'type': type(data[key]), 'length': len(data[key]) if hasattr(data[key], '__len__') else 'N/A'} for key in keys}

# data_info
# print(data_info)

# {'smpl_loss': {'type': <class 'float'>, 'length': 'N/A'},
#  'smpl_poses': {'type': <class 'numpy.ndarray'>, 'length': 720},
#  'smpl_scaling': {'type': <class 'numpy.ndarray'>, 'length': 1},
#  'smpl_trans': {'type': <class 'numpy.ndarray'>, 'length': 720}}

