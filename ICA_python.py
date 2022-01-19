from numpy.lib.npyio import genfromtxt
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
from numpy.core.fromnumeric import mean
np.random.seed(0)
sns.set(rc={'figure.figsize': (11.7, 8.27)})

ALPHA = 1.2
CH_NUM = 16


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean


def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def calculate_new_w(w, X):
    # * 원래 코드
    # w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - \
    #     g_der(np.dot(w.T, X)).mean() * w

    # * 논문 알고리즘 적용
    zmean = np.zeros((3), dtype=X.dtype)
    for i in range(X.shape[1]):
        zTw = np.zeros((3, 3), dtype=X.dtype)
        for j in range(3):
            for k in range(3):
                zTw[k][j] = X[:, i][j] * w[k]
        # zTw = np.matmul(X[:, i].T,  w)
        # 일단 여기 문제임.
        zTw3 = np.matmul(np.matmul(zTw, zTw), zTw)
        # print(zmean)
        # for j in range(3):
        #     zmean[j] += (X[0, i] * zTw3[0][j] + X[1, i] *
        #                  zTw3[1][j] + X[2, i] * zTw3[2][j]) / X.shape[1]
        zmean += (np.matmul(X[:, i],
                  np.matmul(np.matmul(zTw, zTw), zTw)) / X.shape[1])
    w_new = zmean - 3 * w

    return w_new


# 논문 알고리즘 적용
def symm_orth(W):
    # ? NORM2 Largest absolute value
    # max_sum = 0
    # for row in range(W.shape[0]):
    #     temp_max = 0
    #     for col in range(W.shape[1]):
    #         if temp_max < np.abs(W[row][col]):
    #             temp_max = np.abs(W[row][col])
    #     max_sum += temp_max

    # for row in range(W.shape[0]):
    #     temp_max = 0
    #     for col in range(W.shape[1]):
    #         if temp_max < np.abs(W[col][row]):
    #             temp_max = np.abs(W[col][row])
    #     max_sum += temp_max
    # print(max_sum)
    # W /= max_sum
    # ? NORM3

    # W[0] /= np.abs(W[0]).sum()
    # W[1] /= np.abs(W[1]).sum()
    # W[2] /= np.abs(W[2]).sum()

    W = 3/2 * W - 1/2 * np.matmul(np.matmul(W, W.T), W)

    return W


def norm_calc(W):
    # ? NORM1
    # W[0] /= np.abs(W[0]).sum()
    # W[1] /= np.abs(W[1]).sum()
    # W[2] /= np.abs(W[2]).sum()
    # ? NORM2 Largest absolute value
    # max_sum = 0
    # for col in range(W.shape[0]):
    #     if max_sum < np.abs(W[:, col]).sum():
    #         max_sum = np.abs(W[:, col]).sum()
    # W /= max_sum
    # ? NORM2
    # max_sum = 0
    # for col in range(W.shape[0]):
    #     if max_sum < np.abs(W[:, col]).sum():
    #         max_sum = np.abs(W[:, col]).sum()
    # W /= max_sum
    # ? NORM3
    # W[0] /= np.sqrt((W[0]**2).sum())
    # W[1] /= np.sqrt((W[1]**2).sum())
    # W[2] /= np.sqrt((W[2]**2).sum())
    # ? NORM4
    W[0] /= np.sqrt((W[0, :] ** 2).sum())
    W[1] /= np.sqrt((W[1, :] ** 2).sum())
    W[2] /= np.sqrt((W[2, :] ** 2).sum())

    return W


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]

    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    W_new = np.zeros((components_nr, components_nr), dtype=X.dtype)
    W_ica = np.zeros((components_nr, components_nr), dtype=X.dtype)

    # Generate random W (initial value)
    np.random.seed(100)
    for i in range(components_nr):
        W[i] = np.random.rand(components_nr)
    #################################

    for i in range(iterations):
        # one-unit fast ica module
        W_new[0, :] = calculate_new_w(W[0, :], X)
        W_new[1, :] = calculate_new_w(W[1, :], X)
        W_new[2, :] = calculate_new_w(W[2, :], X)
        ############################

        # Error calculation module
        distance = np.abs(np.abs(W_new) - np.abs(W_ica)).sum()

        W_ica = W_new.copy()
        W = W_new.copy()

        W = norm_calc(W)

        if distance < tolerance:
            print("ITERATIONS:")
            print(i+1)
            print("DISTSANCE:")
            print(distance)
            print(np.abs((np.dot(W.T, W) - np.identity(n=3))).sum())
            print("W")
            print(W)
            break
        #############################

        # Symmetric Orthogonalization
        W[0] /= np.sqrt((W[0] ** 2).sum())
        W[1] /= np.sqrt((W[1] ** 2).sum())
        W[2] /= np.sqrt((W[2] ** 2).sum())
        while(1):
            W = symm_orth(W)
            orth_test = np.abs((np.dot(W.T, W) - np.identity(n=3))).sum()
            print(orth_test)
            if orth_test < 0.0001:
                break
        ##############################

        # NORM Divider Block
        W = norm_calc(W)
        ##########################

    # * 원래 코드
    # for i in range(components_nr):

    #     w = np.random.rand(components_nr)

    #     for j in range(iterations):
    #         w_new = calculate_new_w(w, X)
    #         if i >= 1:
    #             w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
    #         # print(W[:i])
    #         distance = np.abs(np.abs((w * w_new).sum()) - 1)
    #         w = w_new
    #         if distance < tolerance:
    #             break
    #     W[i, :] = w
    # print(W)
    S = np.dot(W, X)
    return S


def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")

    fig.tight_layout()
    plt.show()


def plot_mixture_predictions(X, S):
    fig = plt.figure()
    # plt.subplot(2, 1, 1)
    # for x in X:
    #     plt.plot(x)
    # plt.title("mixtures")
    # plt.subplot(2, 1, 2)
    # for s in S:
    #     plt.plot(s)

    plt.plot(S[0])

    plt.title("predicted sources")
    fig.tight_layout()
    plt.show()


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:

            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:

        X += 0.02 * np.random.normal(size=X.shape)

    return X


n_samples = 2000
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)  # sinusoidal
s2 = np.sign(np.sin(3 * time))  # square signal
s3 = signal.sawtooth(2 * np.pi * time)  # saw tooth signal

X = np.c_[s1, s2, s3]
A = np.array(([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]))
# print(A)
X = np.dot(X, A.T)
# X = genfromtxt('PCA.csv', delimiter=',', encoding="utf8", dtype=float)
# print(X)
X = X.T

S = ica(X, iterations=70)
# plot_mixture_predictions(X, S)
# plot_mixture_sources_predictions(X, [s1, s2, s3], np.dot(S.T, A.T).T)
plot_mixture_sources_predictions(X, [s1, s2, s3], S)

# sampling_rate, mix1 = wavfile.read('mix1.wav')
# sampling_rate, mix2 = wavfile.read('mix2.wav')
# sampling_rate, source1 = wavfile.read('source1.wav')
# sampling_rate, source2 = wavfile.read('source2.wav')
# X = mix_sources([mix1, mix2])
# S = ica(X, iterations=1000)
# plot_mixture_sources_predictions(X, [source1, source2], S)
# wavfile.write('out1.wav', sampling_rate, S[0])
# wavfile.write('out2.wav', sampling_rate, S[1])
