import numpy as np
from sklearn.neighbors import KernelDensity
from densityratio.pykliep.pykliep import DensityRatioEstimator


# KDE
def compute_density_ratio(source_data, target_data):
    # get the first variable
    source_x0 = source_data['Area'].values
    target_x0 = target_data['Area'].values

    # Using a Gaussian kernel density estimator
    bandwidth = 0.5  # bandwidth
    kde_s = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(source_x0.reshape(-1, 1))
    kde_t = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(target_x0.reshape(-1, 1))

    # Calculate logarithmic density estimates for source and target data set
    log_density_s = kde_s.score_samples(source_x0.reshape(-1, 1))
    log_density_t = kde_t.score_samples(target_x0.reshape(-1, 1))

    # compute the ratio
    density_ratio = np.exp(log_density_t - log_density_s)
    # print
    return density_ratio


# KLIEP
def compute_density_ratio_kliep(source_data, target_data):
    source_x0 = source_data['Area'].values.reshape(-1, 1)
    source_y0 = source_data['Class'].values.reshape(-1, 1)
    target_x0 = target_data['Area'].values.reshape(-1, 1)
    # the kliep' implementation
    kliep = DensityRatioEstimator()
    kliep.fit(source_x0, target_x0)  # keyword arguments are X_train and X_test
    density_ratio = kliep.predict(source_x0)

    return density_ratio
