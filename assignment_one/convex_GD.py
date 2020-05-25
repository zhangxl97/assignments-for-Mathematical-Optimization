import numpy as np
from copy import copy
import math
import matplotlib.pyplot as plt


class ConvexCase:
    def __init__(self, m, n, seed):
        # initialize A, w, b
        #      shape
        # A   (m, n)
        # w   (n, 1)
        # b   (m, 1)
        np.random.seed(seed)
        self.A = np.random.randn(m, n)
        self.w = np.ones((n, 1))
        self.b = np.dot(self.A, self.w) + np.random.randn(m, 1)

        # error f(w) = ||b - A * w||^2
        self.error = []
        # learning rate: alpha = 1 / L
        # L = max(eig(▽^2(f(w))))  Lipschitz continuity
        # self.learning_rate = 1.0 / np.max(np.linalg.eig(np.dot(self.A.T, self.A))[0])
        self.learning_rate = 0

    def function(self, w):
        # error f(w) = ||b - A * w||^2
        return np.linalg.norm(self.b - np.dot(self.A, w)) ** 2

    def df(self, w):
        # ▽(f(w)) = -2 * A.T * b + 2 * A.T * A * w
        return - 2 * np.dot(self.A.T, self.b) + 2 * np.dot(np.dot(self.A.T, self.A), w)

    def linear_search(self, x1, x2, g, x, s):
        t = (-1 + math.sqrt(5)) / 2
        alpha1 = 0
        while x2 - x1 > 0.001:
            alpha2 = t * (x2 - x1) + x1
            alpha1 = t * t * (x2 - x1) + x1
            g1 = g(x + alpha1 * s)
            g2 = g(x + s * alpha2)
            if g1 < g2:
                x2 = alpha2
            else:
                x1 = alpha1
        return alpha1

    def train(self, stop=1e-3, PRINT=False):
        count = 0
        self.error.append(np.log10(self.function(self.w)))
        s = -self.df(self.w)
        self.learning_rate = self.linear_search(0, 1, self.function, self.w, s)
        w_ans = self.w + self.learning_rate * s
        # while np.linalg.norm(w_ans - self.w) > 0.00001:
        while self.function(self.w) > stop:
            self.w = copy(w_ans)
            self.error.append(np.log10(self.function(self.w)))
            s = -self.df(self.w)
            # w = w - learning_rate * ▽(f(w))
            self.learning_rate = self.linear_search(0, 1, self.function, self.w, s)
            w_ans += self.learning_rate * s
            count += 1

        if PRINT:
            print("＝＝最急降下法＝＝")
            # print(w_ans)
            print(self.function(w_ans))
            print(count)

        return count

    def plot(self):
        plt.plot(self.error, 'r')
        plt.show()


def find_seed():
    seed = 30000
    cnt = 36
    s = 22989
    while seed < 30110:
        convex_case = ConvexCase(5, 20, seed)
        c = convex_case.train(1e-2, False)
        if c > cnt:
            cnt = c
            s = seed
        seed += 1
        # convex_case.plot()

    print(cnt)
    print(s)


def main():
    stop = 1e-2
    convex_case = ConvexCase(100, 200, 22989)
    c = convex_case.train(stop, True)
    convex_case.plot()


if __name__ == '__main__':
    # find_seed()
    main()
