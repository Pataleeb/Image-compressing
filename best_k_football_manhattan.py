import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

class KMeansImpl:
    def __init__(self):
        #data = np.array(image)
        self.pixelno = None
        self.digitno = None
        self.classno = 1
        self.centroids = [3, 6, 12, 24, 48]
        self.original_shape = None
        self.clusters = None
        self.cluster_center = None
        self.cluster_num = None
        pass

    @staticmethod
    def load_image(image_name):
        img=Image.open(image_name)
        return np.array(img)

    def compress(self, pixels, num_clusters, norm_distance=1):

        self.clusters = num_clusters
        self.original_shape = pixels.shape
        pixels = pixels.reshape(-1, 3)

        np.random.seed(33445)
        random_indices = np.random.choice(pixels.shape[0], num_clusters, replace=False)
        self.centroids = pixels[random_indices]

        number_of_iterations = 200

        start_time = time.time()
        i = -1
        for i in range(number_of_iterations):
            if norm_distance == 1:
                distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, axis=2)
            else:
                distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, ord=norm_distance, axis=2)

            self.cluster_num = np.argmin(distances, axis=1)

        centroidn = []
        empty_clusters = []

        for j in range(num_clusters):
            clusters_n = pixels[self.cluster_num == j]
            if len(clusters_n) == 0:
                empty_clusters.append(j)
                centroidn.append(self.centroids[j])
            else:
                centroidn.append(np.mean(clusters_n, axis=0))
            self.centroids = np.array(centroidn)

        compressed_p = self.centroids[self.cluster_num].astype(np.uint8)
        compressed_img = compressed_p.reshape(self.original_shape)

        end_time = time.time()
        time_taken = round(end_time - start_time, 5)

        pass

        map_results = {
            "class": self.clusters,
            "centroid": self.centroids,
            "img": compressed_img,
            "number_of_iterations": i + 1,
            "time_taken": time_taken,

        }

        return map_results
    def calc_wcss(self,pixels):
        pixels = pixels.reshape(-1, 3)
        wcss=np.sum((pixels-self.centroids[self.cluster_num])**2)
        return wcss

image_names=["football.bmp"]
cluster_size = [3, 6, 12, 24, 48]

for image_name in image_names:
    print(f"Processing image:{image_name}")
    kmeans=KMeansImpl()
    image = kmeans.load_image(image_name)

    wcss_val=[]

    for k in cluster_size:
        result = kmeans.compress(image, num_clusters=k)
        wcss=kmeans.calc_wcss(image)
        wcss_val.append(wcss)

        print(f"KMeans with {k} clusters took {result['time_taken']} seconds and converged in {result['number_of_iterations']} iterations for image {image_name}.")

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_size, wcss_val, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Optimal k')
    plt.xticks(cluster_size)
    plt.grid(True)
    plt.show()