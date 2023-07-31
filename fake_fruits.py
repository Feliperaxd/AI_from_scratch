import numpy as np


class FakeFruit:


    def __init__(
        self: 'FakeFruit'        
    ) -> None:

        """
        ----------------------------------------
        Note: Approximate Data
        ----------------------------------------
        COLUMN                             INDEX
        Name.................................0                                           
        Minimum Diameter (cm)................1  
        Maximum Diameter (cm)................2
        Minimum Density (g/cm³)..............3
        Maximum Density (g/cm³)..............4
        Minimum Texture Level (0 ∈ 1)........5
        Maximum Texture Level (0 ∈ 1)........6
        """

        self.fruits_data = [
            ['Apple', 5, 9, 0.6, 0.8, 0.0, 0.3],
            ['Orange', 6, 10, 0.8, 1.2, 0.4, 0.8],
            ['Cherry', 1, 2, 0.9, 1.1, 0.0, 0.2],
            ['lemon', 4, 8, 0.7, 1.0, 0.4, 0.8],
            ['pomegranate', 6, 10, 0.9, 1.2, 0.0, 0.3]
        ]

        self.fruit_index = np.random.randint(
            low=0, 
            high=len(self.fruits_data)
        )
        self.one_hot_vector = np.zeros(len(self.fruits_data)) #  Creates the One-hot vector 
        self.one_hot_vector[self.fruit_index] = 1

        self.data = self.fruits_data[self.fruit_index]
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