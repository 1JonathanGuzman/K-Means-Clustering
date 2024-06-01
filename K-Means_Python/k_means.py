import math
import numpy as np
def euclid_dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def update_centroids(data, cluster_assignments, k):
    centroids = np.zeros((k, data.shape[1]))

    for i in range(k):
        # Find indices of samples assigned to the current cluster
        indices = np.where(cluster_assignments == i)

        # Calculate the mean of samples in the current cluster along axis 0 (rows)
        cluster_mean = np.mean(data[indices], axis=0)

        # Update the centroid for the current cluster
        centroids[i] = cluster_mean

    return centroids

def find_SSE(data, cluster_assignments, k, centroids):
    SSE = 0
    for i in range(k):
        # Find indices of samples assigned to the current cluster
        indices = np.where(cluster_assignments == i)

        # Get the samples belonging to the current cluster
        cluster_samples = data[indices]


        # Calculate SSE for this cluster
        squared_error = 0
        for sample in cluster_samples:
            dist = euclid_dist(sample, centroids[i])
            squared_error += math.pow(dist, 2)
        SSE += squared_error
    return SSE

if __name__ == "__main__":
    seeds = np.loadtxt("seeds.txt")
    MAX_ITERATIONS = 100
    for k in range(3, 8, 2):
        """ Each element will represent a list of samples assigned to that centroid
            The dimensions of clusters will be (x, k), where k is 3, 5, or 7, and x
            represents any number of samples assigned to the column representing that
            centroid
        """
        average_SSE = 0
        total_SSE = 0
        delta_SSE = 1
        last_SSE = 0
        for new_centroid_iteration in range(10):
            centroids = []
            cluster_assignments = np.zeros(seeds.shape[0], dtype=int)
            random_samples = np.random.choice(seeds.shape[0], k, replace=False)
            random_samples = seeds[random_samples]
            # Assign each centroid to represent a cluster
            for sample in random_samples:
                centroids.append(sample)
            convergence_iterations = 0
            while convergence_iterations < MAX_ITERATIONS and delta_SSE >= 0.001:
                # Assign each point to its closest cluster
                sample_num = 0
                for point in seeds:
                    centroid_index = 0
                    closest_centroid_index = 0
                    shortest_dist = 0
                    for centroid in centroids:
                        curr_dist = euclid_dist(point, centroid)
                        if centroid_index == 0:
                            closest_centroid_index = centroid_index
                            shortest_dist = curr_dist
                        elif curr_dist < shortest_dist:
                            closest_centroid_index = centroid_index
                            shortest_dist = curr_dist
                        centroid_index += 1
                    # Assign each sample as an int representing its closest centroid
                    cluster_assignments[sample_num] = closest_centroid_index
                    sample_num += 1
                centroids = update_centroids(seeds, cluster_assignments, k)
                convergence_iterations += 1
                curr_SSE = find_SSE(seeds, cluster_assignments, k, centroids)
                if last_SSE == -99:
                    delta_SSE = find_SSE(seeds, cluster_assignments, k, centroids)
                else:
                    delta_SSE = abs(curr_SSE - last_SSE)
                last_SSE = curr_SSE
            total_SSE += last_SSE
        average_SSE = total_SSE / 10
        # At this point, curr_SSE should have the final SSE value
        print(f"The average SSE for k= {k} is {average_SSE}")
