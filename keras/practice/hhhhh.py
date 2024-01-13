import requests
import pprint
import json

url = 'https://apis.data.go.kr/B551015/API8_2/raceHorseInfo_2?ServiceKey=B%2BllXNbZhSEcdKnh4pIku0fn9non1A%2B0asb12WzeIfaRIxHKRg%2F0eugVvzmIboLegFjB4xqqucZ64AkUABLRvg%3D%3D&pageNo=1&numOfRows=1000'

response = requests.get(url)
contents = response.text



pp = pprint.PrettyPrinter(indent=4)
print(pp.pprint(contents))

# json_ob = json.loads(contents)
# print(json_ob)
# print(type(json_ob)) #json타입 확인


# body = json_ob['response']['body']['items']
# print(body)


