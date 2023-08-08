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
    Minimum Sugar Level (g/100g)..............7
    Maximum Sugar Level (g/100g)..............8
    Minimum PH Level..........................9
    Maximum PH Level.........................10
    """

    fruits_data = [
        ["Pineapple", 10, 20, 0.5, 1.0, 0.7, 0.9, 0.5, 1.0, 2.8, 3.5],
        ["Acerola", 1, 3, 0.8, 1.2, 0.1, 0.2, 0.8, 1.2, 3.5, 4.5],
        ["Açaí", 0.5, 1, 1.1, 1.4, 0.01, 0.1, 1.1, 1.4, 3.0, 4.0],
        ["Plum", 2, 5, 0.9, 1.2, 0.01, 0.1, 0.9, 1.2, 3.8, 4.6],
        ["Cashew", 2, 4, 0.7, 1.1, 0.01, 0.1, 0.7, 1.1, 4.5, 5.5],
        ["Cherry", 1, 2, 0.9, 1.1, 0.01, 0.1, 0.9, 1.1, 3.0, 4.0],
        ["Guava", 5, 10, 0.6, 1.0, 0.2, 0.3, 0.6, 1.0, 3.5, 4.5],
        ["Jabuticaba", 1, 3, 1.0, 1.3, 0.01, 0.1, 1.0, 1.3, 4.0, 5.0],
        ["Kiwi", 5, 8, 0.7, 1.1, 0.4, 0.5, 0.7, 1.1, 3.0, 4.5],
        ["Orange", 7, 10, 0.8, 1.1, 0.3, 0.4, 0.8, 1.1, 3.0, 4.0],
        ["Lemon", 4, 6, 0.9, 1.2, 0.3, 0.4, 0.9, 1.2, 2.0, 3.0],
        ["Apple", 6, 8, 0.8, 1.0, 0.01, 0.1, 0.8, 1.0, 3.5, 4.0],
        ["Mango", 8, 15, 0.7, 1.1, 0.1, 0.2, 0.7, 1.1, 3.4, 4.5],
        ["Passion Fruit", 3, 5, 0.8, 1.2, 0.2, 0.3, 0.8, 1.2, 2.8, 3.7],
        ["Watermelon", 15, 30, 0.3, 0.6, 0.1, 0.2, 0.3, 0.6, 5.5, 6.5],
        ["Melon", 10, 20, 0.5, 0.9, 0.1, 0.2, 0.5, 0.9, 6.0, 7.0],
        ["Blueberry", 0.2, 0.8, 1.2, 1.6, 0.1, 0.2, 1.2, 1.6, 3.5, 4.5],
        ["Pear", 5, 8, 0.8, 1.1, 0.01, 0.1, 0.8, 1.1, 3.0, 4.0],
        ["Peach", 4, 6, 0.8, 1.1, 0.3, 0.4, 0.8, 1.1, 3.3, 4.5],
        ["Surinam Cherry", 1, 3, 0.8, 1.2, 0.2, 0.3, 0.8, 1.2, 3.5, 4.5],
        ["Dragon Fruit", 5, 15, 0.4, 0.8, 0.9, 1.0, 0.4, 0.8, 5.0, 6.0],
        ["Pomegranate", 4, 6, 0.9, 1.1, 0.2, 0.3, 0.9, 1.1, 2.8, 3.8],
        ["Tangerine", 5, 8, 0.8, 1.0, 0.3, 0.4, 0.8, 1.0, 3.8, 4.8],
        ["Tomato", 2, 5, 0.9, 1.3, 0.01, 0.1, 0.9, 1.3, 4.0, 4.8],
        ["Grape", 1, 2, 1.0, 1.2, 0.8, 1.0, 1.0, 1.2, 2.9, 3.8]
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
        self.sugar_level = np.random.uniform(
            low=self.data[7],
            high=self.data[8]
        )
        self.ph_level = np.random.uniform(
            low=self.data[9],
            high=self.data[10]
        )
        self.volume = (4/3) * np.pi * (self.diameter / 2) ** 3
        self.weight = self.volume * self.density
#:)