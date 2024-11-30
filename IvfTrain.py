from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
import random
import os
import numpy as np
number_of_training_data = int(1e4)

class IvfTrain:
    def __init__(self, generated_database = "saved_db.dat", clusters = "", centroids = "", indexes = ""):
        self.generated_database = generated_database
        self.clusters = clusters
        self.centroids = centroids
        self.indexes = indexes
        self.dimension = 70
        self.training_data_number()
        
    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * self.dimension * np.dtype(np.float32).itemsize
            mmap_vector = np.memmap(self.generated_database, dtype=np.float32, mode='r', shape=(1, self.dimension), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def training_data_number(self):
        self.training_data = number_of_training_data
        return self.training_data
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.generated_database) // (self.dimension * np.dtype(np.float32).itemsize)

    def build_index(self):
        #   Here, we should train a sample of the given databse, as
        # training the whole 20 million vectors is very difficult given the constarints

        # print(min(self._get_num_records(), self.training_data)-1)
        # print(len(self.generate_database))
        training_vector_indexes = random.sample(range(self._get_num_records()), min(self._get_num_records(), self.training_data))

        training_vector = [self.get_one_row(row) for row in training_vector_indexes]
        # number if clusters
        k = 10
        chunk_size = self._get_num_records() // k
        print(len(training_vector))
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(training_vector)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        clusters = [[] for i in range(k)]

        for chunk_idx in range(k):
            start_id = chunk_idx * chunk_size
            end_id = start_id + chunk_size
            feature_vectors = [self.get_one_row(row_id) for row_id in range(start_id, end_id)]
            #feature_vectors = [tuple(row[1:]) for row in id_feature]
            # print(len(feature_vectors[99]))
            # print(feature_vectors[0])
            # print(feature_vectors[434])
            # print(id_feature[0])
            # print(feature_vectors[0])
            predictions = kmeans2(feature_vectors, centroids.tolist(), minit="matrix")[1]
            print(predictions.shape)
            for idx, cluster_id in enumerate(predictions):
                clusters[cluster_id].append(feature_vectors[idx])

        # print("Cluster Labels:\n", clusters.shape)
        print("Centroids:\n", centroids.shape)