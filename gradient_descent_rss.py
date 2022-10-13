import numpy as np
from numpy import ndarray

def gradient_descent(actual_values: ndarray,
                     data: ndarray,
                     rate_learning: float = 0.1) -> ndarray:

    # initialization
    # intesection
    b0: float = np.random.rand()
    # cant of independent param
    n_estim: int = np.size(data, 1)
    # random estimators
    estimators: ndarray = np.random.randn(n_estim)
    # sum of estimators * samp array
    est_samp: float = 0
    # cant of samples
    n_samples: int = np.size(data, 0)
    # sum of actual values
    sum_actual: float = np.sum(actual_values)

    # rss
    print(cost_function(actual_values, data, estimators, b0))

    for sample in data:
        est_samp = np.sum(np.multiply(estimators, sample))
        # upd b0
        b0 = b0 - rate_learning* (-1/n_samples * (sum_actual - n_samples*(b0+est_samp)))

        for _ in range(n_estim):
            old_est = estimators[_]
            # upd bi
            estimators[_] = estimators[_] - rate_learning* (-1/n_samples * sample[_] * (sum_actual - n_samples*(b0+est_samp)))
            # update sum with new bi
            est_samp = est_samp - old_est*sample[_] + estimators[_]*sample[_]

        print(estimators)
        print(cost_function(actual_values, data, estimators, b0))
        input()

def cost_function(actual_values: ndarray,
                  data: ndarray,
                  coeff: ndarray,
                  intersection: float) -> float:

    m: int = np.size(actual_values, 0)
    estimated: ndarray = np.add(intersection, np.sum(np.multiply(data, coeff), axis=1))
    return 1/(2*m) * np.sum(np.power(np.subtract(actual_values, estimated),2))
