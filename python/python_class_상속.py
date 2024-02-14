class Animal:
    def __init__(self, name):
        self.name = name
        
        def speak(self):
            pass
        
class Dog(Animal):
    def speak(self):
        return f"{self.name} says 멍멍!"
    
class Cat(Animal):
    def speak(self) :
        return f"{self.name} says 야옹!" 

dog = Dog("멍멍이")       
print(dog.name)
print(dog.speak())        
        
cat = Cat("냥냥이")       
print(cat.name)
print(cat.speak())        
        