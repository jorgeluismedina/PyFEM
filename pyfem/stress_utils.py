
import numpy as np
from scipy import linalg
"""
    Calculate principal stresses from normal stresses using Mohr's circle formulas.

    Args:
    - normal_stresses (numpy.ndarray): Array of normal stresses with shape (3, n) where n is the number of integration points.
                                        Each column contains normal stresses [sigma_x, sigma_y, tau_xy] at an integration point.

    Returns:
    - principal_stresses (numpy.ndarray): Array of principal stresses with shape (3, n) where n is the number of integration points.
                                          Each column contains principal stresses [sigma1, sigma2, dir] at an integration point.
    """
def principal_stresses_mohr(normal_stresses):
    n = normal_stresses.shape[1]
    principal_stresses = np.zeros((3, n))
    for i in range(n):
        sigma_x, sigma_y, tau_xy = normal_stresses[:, i]

        # Calculate principal stresses
        sigma_avg = (sigma_x + sigma_y) / 2
        delta_sigma = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma1 = sigma_avg + delta_sigma
        sigma2 = sigma_avg - delta_sigma

        # Calculate principal direction
        if tau_xy != 0:
            theta_p = np.arctan2(2 * tau_xy, sigma_x - sigma_y) / 2
        else:
            theta_p = 0

        principal_stresses[:, i] = [sigma1, sigma2, theta_p]

    return principal_stresses 
