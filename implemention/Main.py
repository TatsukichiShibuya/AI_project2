import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from catboost import CatBoost, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, data, lr=1):
        print(data)
        self.data = data.copy()
        self.data['bias'] = 1
        self.n, self.d = self.data.shape
        self.param = np.array(np.random.rand(self.d-1), dtype=np.float32)
        self.lr = lr

        print('Train data')
        print(self.data)

    def learn(self):
        print('Start learning')

        start_time = time.time()
        step_count = 0
        best_param = self.param.copy()
        best_correct = 0
        evaluation_curve = []
        flag = True

        while flag:
            if time.time() - start_time > 1:
                break

            flag = False

            train_rand = self.data.iloc[np.random.permutation(
                self.data.index)].reset_index(drop=True)
            df_train = train_rand.drop(columns='label')
            df_label = train_rand['label']

            step_count += 1
            print('Step:', step_count)
            for x, label in zip(df_train.values, df_label):
                y = np.dot(x, self.param)

                if np.sign(y) != np.sign(label):
                    print('Miss')
                    flag = True
                    self.param += x*label*self.lr
                    correct = self.validate(self.param)
                    if correct > best_correct:
                        best_correct = correct
                        best_param = self.param.copy()

            evaluation_curve.append(best_correct)
            print('----------')

        if flag:
            print('Linearly inseparable')
            self.param = best_param
            print('Correct: {}/{}'.format(best_correct, self.n))
        else:
            print('Linearly separable')

        x = np.arange(1, step_count+1)
        plt.plot(x, evaluation_curve)
        plt.show()

    def validate(self, param):
        correct = 0
        for x, label in zip(self.data.drop(columns='label').values, self.data['label']):
            t = np.dot(x, param)
            if np.sign(t) == np.sign(label):
                correct += 1
        return correct

    def show(self):
        print('paramater:', self.param)


def xgboost(data):
    print('-----xgboost-----')
    train = data.drop(columns=['label'])
    label = data['label']
    train_x, test_x, train_y, test_y = train_test_split(
        train, label, test_size=0.2, shuffle=True)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    param = {'max_depth': 10,
             'eta': 0.5,
             'objective': 'binary:logistic',
             'eval_metric': 'logloss'}
    bst = xgb.train(param, dtrain, num_boost_round=100)
    dtest = xgb.DMatrix(test_x, label=test_y)
    pred = bst.predict(dtest)
    pred_class = np.where(pred > 0.5, 1, 0)
    score = accuracy_score(test_y, pred_class)
    print('score:{0:.4f}, {1}/{2}'.format(score,
                                          int(round(len(test_y)*score)),
                                          len(test_y)))
    _, ax = plt.subplots(figsize=(12, 4))
    xgb.plot_importance(bst,
                        ax=ax,
                        importance_type='gain',
                        show_values=False)
    plt.show()


def catboost(data):
    print('-----catboost-----')
    train = data.drop(columns=['label'])
    label = data['label']
    categorical = ['country', 'genre']
    numerical = ['year', 'budget', 'box office',
                 'time', 'star', 'number']
    for col in numerical:
        train[col] = train[col].astype(float)
    train_x, test_x, train_y, test_y = train_test_split(
        train, label, test_size=0.2, shuffle=True)
    train_pool = Pool(data=train_x, label=train_y,
                      cat_features=categorical)
    test_pool = Pool(data=test_x, label=test_y,
                     cat_features=categorical)
    params = {'learning_rate': 0.1,
              'depth': 10,
              'loss_function': 'Logloss',
              'num_boost_round': 1000, }
    cat = CatBoost(params)
    cat.fit(train_pool, eval_set=[test_pool], use_best_model=True)
    pred = cat.predict(test_pool, prediction_type='Class')
    score = accuracy_score(test_y, pred)
    print('score:{0:.4f}, {1}/{2}'.format(score,
                                          int(round(len(test_y)*score)),
                                          len(test_y)))
    feature_importance = cat.get_feature_importance()
    plt.figure(figsize=(12, 4))
    plt.barh(range(len(feature_importance)),
             feature_importance,
             tick_label=train.columns)

    plt.xlabel('importance')
    plt.ylabel('features')
    plt.grid()
    plt.show()


def main(data, flag):

    data.drop(columns=['title'], inplace=True)
    if not flag:
        for col in data.columns:
            if col != 'label':
                data[col] = (data[col]-np.mean(data[col]))/np.std(data[col])

        perceptron = Perceptron(data, lr=0.01)
        perceptron.learn()
        perceptron.show()

        data['label'] = (data['lebal']+1)//2
        xgboost(data)
    else:
        for col in data.columns:
            if not(col in ['label', 'genre', 'country']):
                data[col] = (data[col]-np.mean(data[col]))/np.std(data[col])
        data['label'] = (data['label']+1)//2
        catboost(data)


if __name__ == "__main__":
    path = input('Enter your data directory(If no data, enter sample.csv): ')
    cat_num = input('use categorical data?[y/n]: ')
    data = pd.read_csv(path)
    main(data, flag=(cat_num == 'y'))
