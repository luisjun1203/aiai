def add_numbers(a, b) :
        return a + b
    
result = add_numbers(3,5)
print(result) #출력: 8

class Calculator:
    def __init__(self):
        self.result = 0
        
    def add(self, a, b):
        self.result = a + b
        
    def get_result(self):
        return self.result
        
# 클래스 인스턴스 생성
calc = Calculator()
calc.add(3, 5)
print(calc.get_result())


        