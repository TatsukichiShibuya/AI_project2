import time
import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, data, lr=1):
        self.train = data.drop(columns=['label'])
        self.train['bias'] = 1
        self.label = data['label']
        self.n, self.d = self.train.shape
        self.param = np.ones((self.d,), dtype=np.float32)
        self.lr = lr

        print('Train data')
        print(self.train)

    def learn(self):
        print('Start learning')

        start_time = time.time()
        count = 0
        best_param = self.param.copy()
        best_correct = 0
        flag = True

        while flag:
            if time.time() - start_time > 1:
                break

            flag = False
            count += 1
            correct = 0

            print('Step:', count)
            for x, label in zip(self.train.values, self.label):
                t = np.dot(x, self.param)
                # print('data:{} label:{} param:{} -> t:{}'.format(
                #    x, label, self.param, t))

                if np.sign(t) != np.sign(label):
                    print('Miss')
                    flag = True
                    self.param += x*label*self.lr
                else:
                    correct += 1

            if correct > best_correct:
                best_correct = correct
                best_param = self.param.copy()

            print('----------')

        if flag:
            print('Linearly inseparable')
            self.param = best_param
            print('Correct: {}/{}'.format(best_correct, self.n))
        else:
            print('Linearly separable')

    def show(self):
        print('paramater:', self.param)

    def predict(self, x):
        pass


def kernelFunction(data):
    return data


def main(data):
    feature = kernelFunction(data)
    perceptron = Perceptron(feature, lr=0.01)
    perceptron.learn()
    perceptron.show()


if __name__ == "__main__":
    path = input('Enter your data directory(If no data, enter sample.csv): ')
    data = pd.read_csv(path)
    main(data)
