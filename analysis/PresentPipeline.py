import numpy as np

class PresentPipeline:
    def __init__(self):
        pass

    # data should be a given time window of movement data
    def get_coordination_index(self, data):
        # Determine coordination
        SDV_energy, SVD_axes = self.run_SVD(data)
        bimodality_coefficient = self.run_clustering(data, SVD_axes)

        coordination_index = np.linalg.norm(np.array(1, 0) - np.array(SVD_energy, SVD_axes)) # (SVD, Modality)
    
    # SVD
    # for some given  rank n movement grab the sum energy given by the top n eigenvalues
    # return the sum energy as a percentage of the total energy (0-1) and the top n SVD axes
    def run_SVD(self, data: np.ndarray, rank: int = 1) -> float:
        U, s, Vh = np.linalg.svd(data)

        return np.sum(s[:rank])/np.sum(s), U[:,:rank]
    
    # Cluster across sv axes
    # return the average bimodality coefficient across the top n SVD axes (0-1)
    def run_clustering(self, data: np.ndarray, SVD_axes: np.ndarray, rank: int = 1) -> float:
        avg_bimodality_coefficient = 0

        for axis in SVD_axes:
            projected_data = data @ axis
            bimodality_coefficient = self.get_bimodality_coefficient(projected_data)
            avg_bimodality_coefficient += bimodality_coefficient

        return avg_bimodality_coefficient / len(SVD_axes)

    def get_bimodality_coefficient(self, projected_data: np.ndarray) -> float:
        n = len(projected_data)
        mean = np.mean(projected_data)
        std = np.std(projected_data, ddof=1)  # sample standard deviation
        m3 = np.mean((projected_data - mean) ** 3)

        g1 = m3 / (std ** 3)  # sample skewness (Fisher)
        skewness = g1 * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else 0  # Adjusted Fisher-Pearson standardized moment coefficient

        m4 = np.mean((projected_data - mean) ** 4)
        g2 = m4 / (std ** 4) if std > 0 else 0  # sample kurtosis (Fisher)
        kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * g2
                    - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))) if n > 3 else 0  # Sample excess kurtosis

        bimodality_coefficient = ((skewness ** 2) + 1) / (kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))
        return bimodality_coefficient