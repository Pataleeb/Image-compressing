import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

image_path="football.bmp"
image=Image.open(image_path)

print("Image mode:", image.mode)
print("Image size:", image.size)

class KMeansImpl:
    def __init__(self):
        data = np.array(image)
        self.pixelno = data.shape[2]
        self.digitno = data.shape[0] * data.shape[1]
        self.classno = 1
        self.centroids = [3, 6, 12, 24, 48]
        self.original_shape = None
        self.clusters = None
        self.cluster_center = None
        self.cluster_num = None
        pass

    @staticmethod
    def load_image(image_name="football.bmp"):

        img = Image.open(image_name).convert('RGB')
        return np.array(img)

    def compress(self, pixels, num_clusters, norm_distance=2):

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
            if norm_distance == 2:
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

        # map = {
        # "class": None,
        # "centroid": None,
        # "img": None,
        # "number_of_iterations": None,
        # "time_taken": None,
        # "additional_args": {}
        # }

        map_results = {
            "class": self.clusters,
            "centroid": self.centroids,
            "img": compressed_img,
            "number_of_iterations": i + 1,
            "time_taken": time_taken,
            "additional_args": {}
        }

        return map_results


kmeans = KMeansImpl()
image_array = kmeans.load_image()
cluster_size = [3, 6, 12, 24, 48]
np.random.seed(33445)
for k in cluster_size:
    result = kmeans.compress(image_array, num_clusters=k)
    print(
        f"KMeans with {k} clusters took {result['time_taken']} seconds and converged in {result['number_of_iterations']} iterations.")
    plt.figure(figsize=(5, 5))
    plt.imshow(result["img"])
    plt.title(f"Compressed Image with {k} Clusters")
    plt.axis('off')
    plt.show()
    plt.pause(2)


