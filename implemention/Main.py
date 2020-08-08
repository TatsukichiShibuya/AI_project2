import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, data):
        self.train = data.drop(columns=['label'])
        self.train['bias'] = 1
        print(self.train)
        self.label = data['label']
        self.n, self.d = self.train.shape
        self.param = np.ones((self.d,), dtype=np.float32)

    def learn(self):
        print('start learning')
        # Main process
        flag = True
        while flag:
            flag = False
            for x, label in zip(self.train.values, self.label):
                print(x, label)
                t = np.dot(x, self.param)
                print(t)
                if np.sign(t) != np.sign(label):
                    print('miss')
                    flag = True
                    self.param += x*label
            print('----------')
        # Main process
        print('completed learning')

    def show(self):
        print(self.param)

    def predict(self, x):
        pass


def kernelFunction(data):
    return data


def main(data):
    feature = kernelFunction(data)
    perceptron = Perceptron(feature)
    perceptron.learn()
    perceptron.show()


if __name__ == "__main__":
    path = input('Enter your data directory: ')
    data = pd.read_csv(path)
    main(data)
