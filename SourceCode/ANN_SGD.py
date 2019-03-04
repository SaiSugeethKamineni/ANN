#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# Style guide: pep8
"""
Created on Sat 19 Jan 2019

@author: Sai Sugeeth Kamineni
"""

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
EPOCHS = 100
TMIN = -0.9
TMAX = 0.9
ARCH = [4, 100, 60, 20, 1]
ACTFUN = "relu"


def data_split(file):
    data = open(file, 'r')
    train = open("train.csv", 'w+')
    val = open("validation.csv", 'w+')
    test = open("test.csv", 'w+')

    count = 1

    for i, l in enumerate(data):
        pass

    length = i+1
    data.seek(0)
    for row in data:
        if(count <= round(length*0.9)):
            if(count % 5 != 0):
                train.write(row)
                count += 1
            else:
                val.write(row)
                count += 1
        else:
            test.write(row)

    data.close()
    train.close()
    val.close()
    test.close()


def tanh(x):
    return np.tanh(x)


def tanh_derv(x):
    return 1.0 - np.tanh(x)**2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logsitic_derv(x):
    return logistic(x) * (1 - logistic(x))


def relu(x):
    return x * (x > 0)


def relu_derv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class ANN:
    def __init__(self, arch, actfun):
        self.arch = arch
        self.num_inputs = arch[0]
        self.num_outputs = arch[-1]
        self.num_hidden_layers = len(arch) - 2
        self.hidden_layer_sizes = []
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layer_sizes.append(arch[i+1])
        self.actfun = actfun
        self.weights = []

    def initialise_weights(self):
        for i in range(self.num_hidden_layers + 1):
            if i == 0:
                if self.actfun == "relu":
                    self.weights.append(np.array(np.random.randn(
                        self.arch[i+1], self.arch[i]) * np.sqrt(2 / self.arch[i])))
                else:
                    self.weights.append(np.array(np.random.randn(
                        self.arch[i+1], self.arch[i]) * np.sqrt(1 / self.arch[i])))
            elif i < self.num_hidden_layers:
                if self.actfun == "relu":
                    self.weights.append(np.array(np.random.randn(
                        self.arch[i+1], self.arch[i] + 1) * np.sqrt(2 / self.arch[i])))
                else:
                    self.weights.append(np.array(np.random.randn(
                        self.arch[i+1], self.arch[i] + 1) * np.sqrt(1 / self.arch[i])))
            else:
                self.weights.append(np.array(np.random.randn(
                    self.arch[i+1], self.arch[i] + 1) / np.sqrt(self.arch[i])))
        return self.weights

    def format_input(self, file):
        with open(file) as f:
            f.seek(0)
            rows = f.read().splitlines()
        for i in range(len(rows)):
            rows[i] = rows[i].split(',')
            del rows[i][-1]
        for i in range(len(rows)):
            rows[i] = list(map(float, rows[i]))
        return rows

    def format_target(self, file):
        with open(file) as f:
            rows = f.read().splitlines()
        nvars = len(rows[0].split(',')) - 1
        for i in range(len(rows)):
            rows[i] = rows[i].split(',')
            for j in range(nvars):
                del rows[i][0]
        for i in range(len(rows)):
            rows[i] = list(map(float, rows[i]))
        return rows

    def normalize(self, rows, tmax, tmin):
        unnorm = np.array(rows)
        rmin = min(min(rows))
        rmax = max(max(rows))
        norm = ((unnorm - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return norm

    def add_bias(self, input):
        return np.array(np.insert(input, 0, 1))

    def shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def forward_prop(self, input):
        act_vals = []
        v_vals = []
        prop = []
        act_vals.append(input)
        v_vals.append(input)
        prop.append(input)
        for i in range(self.num_hidden_layers+1):
            temp = prop[-1] @ self.weights[i].T
            v_vals.append(temp)
            if i < self.num_hidden_layers:
                if self.actfun == "relu":
                    temp = relu(temp)
                elif self.actfun == "tanh":
                    temp = tanh(temp)
                elif self.actfun == "logistic":
                    temp = logistic(temp)
            else:
                temp = logistic(temp)
            act_vals.append(temp)
            if i < self.num_hidden_layers:
                temp = self.add_bias(temp)
                prop.append(temp)
        return (v_vals, act_vals, prop)

    def backward_prop(self, v_vals, act_vals, prop, target, learn_rate):
        delWeights = []
        del_j = logsitic_derv(v_vals[-1]) * (target - act_vals[-1])
        del_weight = learn_rate * \
            (del_j.reshape(-1, 1).T @ prop[-1].reshape(1, -1))
        delWeights.append(del_weight)
        for i in reversed(range(1, len(self.weights))):
            if self.actfun == "relu":
                del_j = (relu_derv(v_vals[i]).reshape(
                    1, -1)) * (del_j.reshape(-1, 1).T @ np.delete(self.weights[i], 0, 1))
            elif self.actfun == "logistic":
                del_j = logsitic_derv(
                    v_vals[i]).reshape(1, -1) * del_j.reshape(-1, 1) @ np.delete(self.weights[i], 0, 1)
            else:
                del_j = tanh_derv(
                    v_vals[i]).reshape(1, -1) * del_j.reshape(-1, 1) @ np.delete(self.weights[i], 0, 1)
            del_weight = learn_rate * \
                del_j.reshape(-1, 1) @ prop[i-1].reshape(1, -1)
            delWeights.append(del_weight)
        delWeights.reverse()
        return delWeights

    def train(self, learn_rate, epochs):
        self.initialise_weights()
        train_inputs = self.format_input("train.csv")
        train_targets = self.format_target("train.csv")
        train_norminputs = self.normalize(train_inputs, TMIN, TMAX)
        train_normtargets = self.normalize(train_targets, TMIN, TMAX)
        min = 1
        max = 0
        avg_cost_list = []
        val_cost_list = []
        for epoch in range(epochs):
            count = 0
            cum_cost = 0
            for i in train_norminputs:
                state = self.forward_prop(i)
                cum_cost += np.asscalar(
                    train_normtargets[count] - state[1][-1])**2 / 2
                del_w = self.backward_prop(
                    state[0], state[1], state[2], train_normtargets[count], learn_rate)
                for i in range(len(self.weights)):
                    self.weights[i] += del_w[i]
                count += 1
            avg_cost = cum_cost / count
            self.shuffle_in_unison(train_norminputs, train_normtargets)
            val_cost = self.test("validation.csv")
            if val_cost < min:
                min = val_cost
            if val_cost > max:
                max = val_cost
            if avg_cost < min:
                min = avg_cost
            if avg_cost > max:
                max = avg_cost
            avg_cost_list.append(avg_cost)
            val_cost_list.append(val_cost)
            plt.ylim([min*0.9, max*1.1])
            plt.xlim([0, epochs])
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.title("Training and Validation error plots")
            plt.plot([x for x in range(epoch+1)], avg_cost_list,
                     'r-', label="Training error")
            plt.plot([y for y in range(epoch+1)], val_cost_list,
                     'b-', label="Validation error")
            if epoch == 0:
                plt.legend()
            plt.pause(10**-10)

        return 0

    def predict(self, test_input):
        test_state = self.forward_prop(test_input)
        return test_state[1][-1]

    def test(self, test_data):
        cum_cost = 0
        test_inputs = ann.format_input(test_data)
        test_targets = ann.format_target(test_data)
        test_norminputs = ann.normalize(test_inputs, TMIN, TMAX)
        test_normtargets = ann.normalize(test_targets, TMIN, TMAX)
        count = 0
        for i in test_norminputs:
            cum_cost += np.asscalar(
                test_normtargets[count] - ann.predict(i))**2 / 2
            count += 1
        avg_cost = cum_cost / count
        return(avg_cost)


if __name__ == "__main__":
    data_split("data.csv")
    ann = ANN(ARCH, ACTFUN)
    ann.train(LEARNING_RATE, EPOCHS)
    print("Test error", ann.test("test.csv"))
