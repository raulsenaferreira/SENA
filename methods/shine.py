import cv2
from pywt import dwt2
from scipy import spatial
import numpy as np
import skimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy
from scipy.signal import convolve2d
import math
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def get_threshold(id_dataset_name, range_type='mean'):
    if id_dataset_name == 'cifar10':
        # values of pdf were multplied by 100 just to not be too small to write here
        scale = 100
        if range_type == 'mean':
            return np.array([0.017179, 0.007721, 0.010024, 0.008848, 0.040620, 0.019044, 0.007235, 0.018359, 0.010600,
                             0.008228]) / scale
        elif range_type == 25:
            return np.array([0.008971, 0.003890, 0.004986, 0.004611, 0.019319, 0.010346, 0.003872, 0.009917, 0.005827,
                             0.004431]) / scale
        elif range_type == 75:
            return np.array([0.024920, 0.011284, 0.015168, 0.012670, 0.059786, 0.027257, 0.010531, 0.026183, 0.015281,
                             0.011942]) / scale


def convert_to_grayscale(X):
    return np.dot(X[..., :3], [0.299, 0.587, 0.114])


def measure_energy(img):
    _, (cH, cV, cD) = dwt2(img.T, 'db1')

    return (cH ** 2 + cV ** 2 + cD ** 2).sum() / img.size


def estimate_noise(img):
    # Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996

    H, W = np.shape(img)

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

    return sigma


def create_vector_shine(img, image_features=None):
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)

    H = np.squeeze(graycoprops(glcm, prop='homogeneity'))
    I = skimage.measure.shannon_entropy(np.squeeze(glcm))
    N = estimate_noise(img)
    E = measure_energy(img)  # np.squeeze(graycoprops(glcm, prop='energy'))

    return np.array([H, I, N, E])


def create_matrix_shine(X):
    # comment the 2 lines below if data is not in pytorch tensor format
    n_samples, ndim, nx, ny = X.shape
    X = X.reshape((n_samples, nx, ny, ndim))

    dim = np.shape(X)[3]  # numpy format

    if dim == 3:
        X = convert_to_grayscale(X).astype(np.uint8)
    elif dim == 1:
        nsamples, nx, ny, ndim = X.shape
        X = X.reshape((nsamples, nx * ny, ndim)).astype(np.uint8)

    # data_variance = np.var(x_scaled / 255.0)
    shine_matrix = []

    for img in X:
        shine = create_vector_shine(img)
        shine_matrix.append(shine)

    return np.array(shine_matrix)


def create_vector_energy(X):
    vector_energy = []

    for img in X:
        E = measure_energy(img)
        vector_energy.append(E)

    return np.array(vector_energy)


def calculate_density(X, use_grid_search=True, CV_num=10, bandwidth=1.0):
    if use_grid_search:

        if len(X) <= CV_num:
            CV_num = len(X)

        print("searching the best bandwidth")
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=CV_num, n_jobs=-1)
        grid.fit(X)

        return grid.best_estimator_  # kde estimator
    else:
        kde = KernelDensity(bandwidth=bandwidth).fit(X)
        return kde


# experimenting new thresholds for shine
def run_SHINE_2(monitor_shine, X, id_y_train, pred_train, pred, incoming_feature, threshold_SMimg, threshold_SMout,
                arr_id_threshold, arr_ood_threshold, features):
    # all labels from class c
    ind_y_c = np.where(id_y_train == pred)[0]
    # print('len all labels class',c, np.shape(ind_y_c), ind_y_c)

    # all pred as c
    ind_ML_c = np.where(pred_train == pred)[0]
    # print('len pred class',c, np.shape(ind_ML_c), ind_ML_c)

    # features from correct pred
    ind = set(ind_y_c).intersection(ind_ML_c)
    f_c_correct = features[list(ind)]

    scores = []
    for c in f_c_correct:
        cosine_similarity = 1 - spatial.distance.cosine(c, incoming_feature)
        scores.append(cosine_similarity)

    len_scores = len(scores)
    sorted_scores = sorted(scores)
    ind_threshold = int(len_scores * threshold_SMout)
    min_threshold_sim = sorted_scores[-ind_threshold]
    max_threshold_sim = sorted_scores[ind_threshold]
    # avg_sim = np.sum(scores)/len(scores)

    if max_threshold_sim >= arr_id_threshold:
        return False  # it is ID and correct pred
    elif min_threshold_sim <= arr_ood_threshold:
        return True  # it is OOD
    else:
        # it is not OOD but we do not know if the prediction is correct
        monitor_pred, pdf = monitor_shine.predict(X, pred, threshold_SMimg)
        return monitor_pred
