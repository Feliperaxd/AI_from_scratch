import numpy as np

class ActivationFunctions:


    @staticmethod
    def sigmoid(
        inputs: np.ndarray
    ) -> np.ndarray:

        return np.exp(-inputs) / ((np.exp(-inputs) + 1) ** 2)

    @staticmethod
    def relu(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.maximum(0, inputs)
        return outputs

    @staticmethod
    def tanh(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.tanh(inputs)
        return outputs

    @staticmethod
    def softmax(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.exp(inputs) / np.sum(np.exp(inputs))
        return outputs
#:)