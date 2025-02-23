import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

image_path="sea-turtle-400x225.png"
image=Image.open(image_path)

print("Image mode:", image.mode)
print("Image size:", image.size)
class KMeansImpl:
    def __init__(self):
        data = np.array(image)
        self.pixelno = data.shape[2]
        self.digitno = data.shape[0] * data.shape[1]
        self.classno = 1
        self.centroids = [3,6,12,24,48]
        self.original_shape=None
        self.clusters=None
        self.cluster_center=None
        self.cluster_num=None
        pass
    @staticmethod
    def load_image(image_name="sea-turtle-400x225.png"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        img=Image.open(image_name).convert('RGB')
        return np.array(img)

    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).

        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """

        self.clusters=num_clusters
        self.original_shape = pixels.shape
        pixels = pixels.reshape(-1,3)

        np.random.seed(33445)
        random_indices = np.random.choice(pixels.shape[0], num_clusters, replace=False)
        self.centroids = pixels[random_indices]

        number_of_iterations = 200

        start_time = time.time()
        i=-1
        for i in range(number_of_iterations):
            if norm_distance == 2:
                distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, axis=2)
            else:
                distances = np.linalg.norm(pixels[:, np.newaxis] - self.centroids, ord=norm_distance, axis=2)

            self.cluster_num=np.argmin(distances, axis=1)

        centroidn=[]
        empty_clusters=[]

        for j in range(num_clusters):
            clusters_n=pixels[self.cluster_num==j]
            if len(clusters_n)==0:
                empty_clusters.append(j)
                centroidn.append(self.centroids[j])
            else:
                centroidn.append(np.mean(clusters_n, axis=0))
            self.centroids=np.array(centroidn)

        compressed_p = self.centroids[self.cluster_num].astype(np.uint8)
        compressed_img = compressed_p.reshape(self.original_shape)

        end_time=time.time()
        time_taken=round(end_time-start_time,5)


        pass


        #map = {
            #"class": None,
           # "centroid": None,
            #"img": None,
            #"number_of_iterations": None,
            #"time_taken": None,
            #"additional_args": {}
        #}

        map_results={
            "class": self.clusters,
            "centroid": self.centroids,
            "img":compressed_img,
            "number_of_iterations":i +1,
            "time_taken":time_taken,
            "additional_args":{}
        }

        return map_results

kmeans = KMeansImpl()
image_array = kmeans.load_image()
cluster_size=[3,6,12,24,48]
for k in cluster_size:
    result = kmeans.compress(image_array, num_clusters=k)
    print(
        f"KMeans with {k} clusters took {result['time_taken']} seconds and converged in {result['number_of_iterations']} iterations.")
    plt.figure(figsize=(5, 5))
    plt.imshow(result["img"])
    plt.title(f"Compressed Image with {k} Clusters")
    plt.axis('off')
    plt.show()
    plt.pause(2)  # Pause for 2 seconds to view each image
