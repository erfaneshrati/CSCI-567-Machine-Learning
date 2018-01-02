import json
import random
import numpy as np
from scipy.stats import multivariate_normal as mvn

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    X = np.array(X)
    n, _ = X.shape
    pi = []
    mu = []
    cov = np.zeros((K, 2, 2))
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov[k] = temp_cov

    k = K
    for i in range(100):
        # Expected step
        ws = np.zeros((k, n))
        for j in range(len(mu)):
            for i in range(n):
                ws[j, i] = pi[j] * 1/(2*np.pi*np.sqrt(cov[j][0][0]*cov[j][1][1] - cov[j][1][0]*cov[j][0][1])) * np.exp(-0.5 * np.dot(np.dot((X[i]-mu[j]).T, np.linalg.inv(cov[j])),(X[i]-mu[j])))
        ws /= ws.sum(0)

        # Maximization step 
        pi = np.zeros(k)
        for j in range(len(mu)):
            for i in range(n):
                pi[j] += ws[j, i]
        pi /= n

        mu = np.zeros((k, 2))
        for j in range(k):
            for i in range(n):
                mu[j] += ws[j, i] * X[i]
            mu[j] /= ws[j, :].sum()
        
        cov = np.zeros((k, 2, 2))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mu[j], (2,1))
                cov[j] += ws[j, i] * np.dot(ys, ys.T)
            cov[j] /= ws[j,:].sum()
    cov_new = []
    for i in range(K):
        cov_new.append(cov[i].reshape(4).tolist())
    cov = cov_new
    return mu.tolist(), cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
