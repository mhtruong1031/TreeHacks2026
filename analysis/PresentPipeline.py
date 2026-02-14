import numpy as np

class PresentPipeline:
    def __init__(self):
        pass

    # data should be a given time window of movement data (n_timesteps, n_features)
    def get_coordination_index(self, data, rank: int = 1):
        # Determine coordination
        SVD_energy, SVD_axes = self.run_SVD(data, rank=rank)
        bimodality_coefficient = self.run_clustering(data, SVD_axes)

        coordination_index = np.linalg.norm(np.array([1.0, 0.0]) - np.array([SVD_energy, bimodality_coefficient]))
        return coordination_index
    
    # SVD
    # for some given rank n movement grab the sum energy given by the top n eigenvalues
    # return the sum energy as a percentage of the total energy (0-1) and the top n SVD axes (right singular vectors, shape (n_features, rank))
    def run_SVD(self, data: np.ndarray, rank: int = 1):
        U, s, Vh = np.linalg.svd(data, full_matrices=False)
        energy_ratio = np.sum(s[:rank]) / np.sum(s) if np.sum(s) > 0 else 0.0
        # Right singular vectors (principal directions in feature space)
        SVD_axes = Vh.T[:, :rank]
        return energy_ratio, SVD_axes
    
    # Cluster across sv axes
    # return the average bimodality coefficient across the top n SVD axes (0-1)
    def run_clustering(self, data: np.ndarray, SVD_axes: np.ndarray) -> float:
        n_axes = SVD_axes.shape[1]
        if n_axes == 0:
            return 0.0
        avg_bimodality_coefficient = 0.0
        for i in range(n_axes):
            axis = SVD_axes[:, i]
            projected_data = data @ axis
            bimodality_coefficient = self.get_bimodality_coefficient(projected_data)
            avg_bimodality_coefficient += bimodality_coefficient
        return avg_bimodality_coefficient / n_axes

    def get_bimodality_coefficient(self, projected_data: np.ndarray) -> float:
        n = len(projected_data)
        if n <= 3:
            return 0.0
        mean = np.mean(projected_data)
        std = np.std(projected_data, ddof=1)  # sample standard deviation
        if std == 0:
            return 0.0
        m3 = np.mean((projected_data - mean) ** 3)
        g1 = m3 / (std ** 3)  # sample skewness (Fisher)
        skewness = g1 * np.sqrt(n * (n - 1)) / (n - 2)  # Adjusted Fisher-Pearson standardized moment coefficient

        m4 = np.mean((projected_data - mean) ** 4)
        g2 = m4 / (std ** 4)  # sample kurtosis (Fisher)
        kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * g2
                    - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))  # Sample excess kurtosis

        denom = kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        bimodality_coefficient = ((skewness ** 2) + 1) / denom if denom > 0 else 0.0
        return bimodality_coefficient