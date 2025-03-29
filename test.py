class Animal:
    def __init__(self, species, age):
        self.aspecies = species
        self.aage = age

class Dog(Animal):
    def __init__(self, name, species="Cat",age="1"):
        super().__init__(species, age)  # 调用父类的 __init__
        self.name = name  # 绑定自己的属性

# 创建 Dog 对象

dog = Dog("Buddy")

# 访问属性
print(dog.name)     # 输出: Buddy
print(dog.aspecies)  # 输出: Dog
