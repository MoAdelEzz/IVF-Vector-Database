from sklearn.cluster import MiniBatchKMeans
import time
import os
import numpy as np
import struct
from pathlib import Path

number_of_training_data = int(1e4)
DIMENSION = 70

class IvfTrain:
    def __init__(self, generated_database = "saved_db.dat", clusters = "saved_clusters.dat", centroids = "saved_centroids.dat", indexes = "saved_indexes.dat"):
        self.generated_database = generated_database
        self.clusters = clusters
        self.centroids = centroids
        self.indexes = indexes
        self.dimension = 70
        self.build_index()
        
    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * self.dimension * np.dtype(np.float32).itemsize
            mmap_vector = np.memmap(self.generated_database, dtype=np.float32, mode='r', shape=(1, self.dimension), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.generated_database, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def training_data_number(self):
        self.training_data = number_of_training_data
        return self.training_data
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.generated_database) // (self.dimension * np.dtype(np.float32).itemsize)

    def process_clusters_and_centroids(self, centroids, clusters):
        centroid_records = []
        for idx, vector in enumerate(centroids):
            centroid_records.append({"id": idx, "embed": vector})
    
        file = open(self.centroids, 'ab')
        try:
            for record in centroid_records:
                record_id, embedding = record["id"], record["embed"]
                binary_record = struct.pack(f'i{self.dimension}f', record_id, *embedding)
                file.write(binary_record)

        finally:
            file.close()
            del file

    def file_output(self, clusters, centroids):
        Path(self.clusters).touch()
        Path(self.centroids).touch()
        Path(self.indexes).touch()
        for idx, vector in enumerate(clusters):
            file = open(self.clusters, 'ab')
            start_offset, end_offset = None, None
            try:
                start_offset = file.tell()
                
                for row in vector:
                    binary_data = struct.pack('ii', idx, row)
                    file.write(binary_data)
                end_offset = file.tell()
            finally:
                file.close()
                del file
            file = open(self.indexes, 'ab')
            try:
                binary_data = struct.pack('iii', idx, start_offset, end_offset)
                file.write(binary_data)
            finally:
                file.close()
                del file
        self.process_clusters_and_centroids(centroids, clusters)
        
        
            

    def build_index(self):
        start_time = time.time()
        n_clusters = 10000

        mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000)
        vectors = self.get_all_rows()
        mbk.fit(vectors)
        end_time_meh = time.time()
        print(f"Done, Time Spent: {end_time_meh-start_time:.6f} Seconds")

        centroids = mbk.cluster_centers_
        clusters_ids = mbk.labels_

        clusters = [[] for i in range(n_clusters)]
        for idx, cluster_id in enumerate(clusters_ids):
            clusters[int(cluster_id)].append(int(idx))

        print("Training Done, Writing Results....")
        self.file_output(clusters, centroids)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time spent: {elapsed_time:.6f} seconds")