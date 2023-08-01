import numpy as np


class FruitsData:

    """
    ---------------------------------------------
    Note: Approximate data obtained by GPT
    ---------------------------------------------
    COLUMN                                  INDEX
    Name......................................0                                           
    Minimum Diameter (cm).....................1  
    Maximum Diameter (cm).....................2
    Minimum Density (g/cm³)...................3
    Maximum Density (g/cm³)...................4
    Minimum Texture Level (0 ∈ 1).............5
    Maximum Texture Level (0 ∈ 1).............6
    Flavor (0 = Neutral, 1 = Sweet, 2 = Sour).7
    """

    fruits_data = [
        ["Pineapple", 10, 20, 0.5, 1.0, 0.7, 0.9, ],
        ["Acerola", 1, 3, 0.8, 1.2, 0.1, 0.2],
        ["Açaí", 0.5, 1, 1.1, 1.4, 0.0, 0.1],
        ["Plum", 2, 5, 0.9, 1.2, 0.0, 0.1],
        ["Cashew", 2, 4, 0.7, 1.1, 0.0, 0.1],
        ["Cherry", 1, 2, 0.9, 1.1, 0.0, 0.1],
        ["Guava", 5, 10, 0.6, 1.0, 0.2, 0.3],
        ["Jabuticaba", 1, 3, 1.0, 1.3, 0.0, 0.1],
        ["Kiwi", 5, 8, 0.7, 1.1, 0.4, 0.5],
        ["Orange", 7, 10, 0.8, 1.1, 0.3, 0.4],
        ["Lemon", 4, 6, 0.9, 1.2, 0.3, 0.4],
        ["Apple", 6, 8, 0.8, 1.0, 0.0, 0.1],
        ["Mango", 8, 15, 0.7, 1.1, 0.1, 0.2],
        ["Passion Fruit", 3, 5, 0.8, 1.2, 0.2, 0.3],
        ["Watermelon", 15, 30, 0.3, 0.6, 0.1, 0.2],
        ["Melon", 10, 20, 0.5, 0.9, 0.1, 0.2],
        ["Blueberry", 0.2, 0.8, 1.2, 1.6, 0.1, 0.2],
        ["Pear", 5, 8, 0.8, 1.1, 0.0, 0.1],
        ["Peach", 4, 6, 0.8, 1.1, 0.3, 0.4],
        ["Surinam Cherry", 1, 3, 0.8, 1.2, 0.2, 0.3],
        ["Dragon Fruit", 5, 15, 0.4, 0.8, 0.9, 1.0],
        ["Pomegranate", 4, 6, 0.9, 1.1, 0.2, 0.3],
        ["Tangerine", 5, 8, 0.8, 1.0, 0.3, 0.4],
        ["Tomato", 2, 5, 0.9, 1.3, 0.0, 0.1],
        ["Grape", 1, 2, 1.0, 1.2, 0.8, 1.0]
    ]

class FakeFruit:


    def __init__(
        self: 'FakeFruit'        
    ) -> None:

        self.fruit_index = np.random.randint(
            low=0, 
            high=len(FruitsData.fruits_data)
        )
        self.one_hot_vector = np.zeros(len(FruitsData.fruits_data)) #  Creates the One-hot vector 
        self.one_hot_vector[self.fruit_index] = 1

        self.data = FruitsData.fruits_data[self.fruit_index]
        self.name = self.data[0]

        self.diameter = np.random.uniform(
            low=self.data[1],
            high=self.data[2]
        )
        self.density = np.random.uniform(
            low=self.data[3],
            high=self.data[4]
        )
        self.texture = np.random.uniform(
            low=self.data[5],
            high=self.data[6]
        )
        self.volume = (4/3) * np.pi * (self.diameter / 2) ** 3
        self.weight = self.volume * self.density
#:)