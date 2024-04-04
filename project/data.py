import requests

def download_video(url, folder):
    """주어진 URL에서 비디오를 다운로드하여 지정된 폴더에 저장합니다."""
    try:
        response = requests.get(url, stream=True)
        filename = url.split('/')[-1]
        path = f"{folder}/{filename}"
        
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # 필터링된 chunk 데이터를 파일에 쓴다
                    file.write(chunk)
        print(f"{filename} 다운로드 완료.")
    except Exception as e:
        print(f"{url} 다운로드 실패: {e}")

# 다운로드할 비디오가 있는 파일 경로
file_path = 'c:\\_data\\main_project\\refined_10M_all_video_url.csv'
# 비디오를 저장할 폴더
save_folder = 'c:\\_data\\main_project\\motion_data\\'

# 파일에서 URL을 읽어와서 다운로드 실행
with open(file_path, 'r', encoding='utf-8') as file:
    for url in file:
        download_video(url.strip(), save_folder)