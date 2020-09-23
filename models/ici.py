import math
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


def plot_regularization_path(alpha, coefs, query_y, pseudo_y):
    coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, :, :]), axis=2)  # 100, 80
    alpha = np.log10(alpha[::-1])  # 100
    plt.clf()
    plt.figure(1)
    for i in range(5):
        plt.plot(alpha, coefs[:, i], c="y")
    for i in range(5, 80):
        t = i - 5
        c = "r" if query_y[t] == pseudo_y[t] else "black"
        plt.plot(alpha, coefs[:, i], c=c, linestyle="-")
    plt.savefig(f"./temp/path_{time.time()}.png")


class ICI(object):

    def __init__(self, classifier='lr', num_class=None, step=5, max_iter='auto', reduce='pca', d=5, norm='l2'):
        self.step = step  # the number of unlabeled data selected for each class in one iteration
        self.max_iter = max_iter
        self.num_class = num_class  # the number of classes per task
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.initial_classifier(classifier)
        self.elastic_net = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                      normalize=True, warm_start=True, selection='cyclic')

    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y

    def predict(self, X, unlabeled_X=None, show_detail=False, query_y=None, disable_ici=False):
        """
        1. Train a liner classifier with support set.
        2. Use the classifier to get pseudo label for unlabeled data.
        3. Choose a subset from the unlabeled data which has high credibility inferred by the ICI algorithm
           and expand this subset to the support set.
        4. Go to step 1 until converged or maximum number of iterations reached.
        """
        support_X, support_y = self.support_X, self.support_y
        way, num_support = self.num_class, len(support_X)
        query_X = self.norm(X)
        if unlabeled_X is None:
            unlabeled_X = query_X
        else:
            unlabeled_X = self.norm(unlabeled_X)
        num_unlabeled = unlabeled_X.shape[0]
        assert self.support_X is not None
        embeddings = np.concatenate([support_X, unlabeled_X])
        X = self.embed(embeddings)
        H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabeled
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabeled / self.step)
        else:
            assert float(self.max_iter).is_integer()
        support_set = np.arange(num_support).tolist()
        # Train classifier using (Xs; ys);
        self.classifier.fit(self.support_X, self.support_y)
        acc_list = []
        if not disable_ici:
            for _ in range(self.max_iter):
                if show_detail:
                    predicts = self.classifier.predict(query_X)
                    acc_list.append(np.mean(predicts == query_y))
                # Get pseudo-label yu for Xu by classifier
                pseudo_y = self.classifier.predict(unlabeled_X)
                y = np.concatenate([support_y, pseudo_y])
                Y = self.label2onehot(y, way)
                y_hat = np.dot(X_hat, Y)
                # Use the ICI algorithm to expand our support set.
                support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, y, query_y)
                y = np.argmax(Y, axis=1)
                self.classifier.fit(embeddings[support_set], y[support_set])
                if len(support_set) == len(embeddings):
                    break
        predicts = self.classifier.predict(query_X)
        if show_detail:
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts

    def expand(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, targets, query_y):
        """
        Here is the key of the ICI algorithm.
        """
        # Check: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet.path
        # Coefficients along the path with shape: (n_features, n_alphas)
        alphas, coefs, _ = self.elastic_net.path(X_hat, y_hat, l1_ratio=1.0)
        # self.plot_regularization_path(alphas, coefs, pseudo_y, query_y)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, num_support:, :]), axis=2)
        selected = np.zeros(way)
        for gamma in coefs:
            # Rank (X; y) = (X; [ys; yu]) by ICI;
            for i, g in enumerate(gamma):
                # Select a subset (X_sub; y_sub) into (Xs; ys);
                if g == 0.0 and (i + num_support not in support_set) and (selected[pseudo_y[i]] < self.step):
                    support_set.append(i + num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
        return support_set

    # Copied from: https://github.com/Yikai-Wang/ICI-FSL/issues/4

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['isomap', 'itsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'itsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5, eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear', probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
