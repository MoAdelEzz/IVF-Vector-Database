from typing import Dict, List, Annotated
import numpy as np
import os
import struct
import gc
from IvfTrain import IvfTrain

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path =  ["saved_clusters.dat", "saved_centroids.dat", "saved_indexes.dat"], new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.cluster_path = index_file_path[0]
        self.centroid_path = index_file_path[1]
        self.index_path = index_file_path[2]
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"
    

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def get_all_rows_values(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors[:,1:])
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        
        centroids = []
        file = open("saved_centroids.dat", 'rb')
        try:
            row_size = ELEMENT_SIZE * (DIMENSION + 1)
            data = file.read()
            length = len(data)
            for offset in range(0, length, row_size):
                packed_data = data[offset:offset + row_size]
                if len(packed_data) < row_size:
                    break
                unpacked_data = struct.unpack(f'i{DIMENSION}f', packed_data)
                del packed_data
                
                centroids.append(unpacked_data)
            
        finally:
            file.close()
            del file
            
        centroids = np.array(centroids)
        sorted_indices = np.argsort((np.dot(centroids[:, 1:], query.T).T / (np.linalg.norm(centroids[:, 1:], axis=1) * np.linalg.norm(query))).squeeze())[::-1]

        best_centroids = sorted_indices.tolist()

        del sorted_indices
        
        scores = best_centroids[:4]
        del best_centroids
        top_k_results = []
        for score in scores:
            first_index, second_index = None, None
            file = open("saved_indexes.dat", 'rb')
            try:
                position = 3 * score * ELEMENT_SIZE
                file.seek(int(position))
                packed_data = file.read(3 * ELEMENT_SIZE)
                unpacked_data = struct.unpack('iii', packed_data)
                del packed_data
                first_index, second_index = unpacked_data[1], unpacked_data[2]
                
            finally:
                file.close()
                del file
                
            ranged_clusters_ids = []
            with open("saved_clusters.dat", 'rb') as file:
                file.seek(first_index)
                while file.tell() < second_index:
                    packed_data = file.read(2 * ELEMENT_SIZE)
                    if packed_data == b'':
                        break
                    data = struct.unpack('ii', packed_data)
                    del packed_data
                    ranged_clusters_ids.append(data)
                    
                file.close()
                del file
                
            
            ranged_clusters = [(self.get_one_row(id[1]),id[1]) for id in ranged_clusters_ids]
            
            cosine_similarities = []
            for row in ranged_clusters:
                cosine_similarity = self._cal_score(query, row[0])
                cosine_similarities.append((cosine_similarity, row[1]))
            
            top_k_results.extend(cosine_similarities)
            del cosine_similarities
            
        print(len(top_k_results))
        scores = sorted(top_k_results, key=lambda x: x[0], reverse=True)[:top_k]
        gc.collect()
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2).T
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        IvfTrain()