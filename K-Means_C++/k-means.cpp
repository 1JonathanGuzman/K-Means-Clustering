#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <time.h> // time
#include <stdlib.h> // rand, srand

class DimensionUniformityError : public std::exception {
    public:
    const char * what (){
        return "An exception has been thrown: DimensionUniformityError. See error log for more details.";
    }
};

double find_SSE(std::vector<std::vector<double>> dataset, std::vector<int> cluster_assignment, std::vector<std::vector<double>> centroids);
void update_centroids(std::vector<std::vector<double>> dataset, std::vector<int> cluster_assignment, std::vector<std::vector<double>>& centroids, int num_centroids);


double euclid_dist(std::vector<double> p, std::vector<double> q) {
    /* Function that takes in points p and q from a dataset and
    returns their euclidean distance as a double. Points p and q
    can have any number of dimensions, but p and q must have the
    same dimensions. Otherwise, this function will throw an error. */

    if (p.size() != q.size()){
        std::cerr << "Error in euclid_dist: The vectors (points) passed ";
        std::cerr << "are not of equal size (dimensions).\n";
        DimensionUniformityError DUE;
        throw DUE;
    }
    int N = p.size(); // N represents: number of dimensions

    /* Logic for euclidean distance for N dimensions is in the for loop below:
    euclid_dist = square root of sum squared of elements p(i) and q(i) where i is the
    current element, and ranges from 0 to N. */
    double dist = 0;
    for(int i = 0; i < N; i++){
        dist += pow(p.at(i) - q.at(i), 2);
    }
    return sqrt(dist);
}

std::vector<std::vector<double>> minmax_normalize(const std::vector<std::vector<double>> dataset){
    /* Function that takes in a dataset, defined as a vector
    of std::vector<double> items. In essence, a 2D array to
    represent a dataset of N dimensions. Dimension uniformity
    must be enforced by the user before passing to this function.
    All records must have the same amount of features.*/

    /* min_and_max: Vector with size equal to number of dimensions, containing
    two values per dimension: one representing the minimum value
    present in the dimension, the other representing the maximum*/
    std::vector<std::vector<double>> min_and_max(dataset[0].size(), std::vector<double>(2, -1));

    /* normalized_dataset: Vector representing normalized version of 'dataset'.
    This will be returned and the original vector will remain unchanged.*/
    std::vector<std::vector<double>> normalized_dataset(dataset.size(),
                                        std::vector<double>(dataset[0].size(), -1));

    for(int i = 0; i < dataset[0].size(); i++){
        /* Unlike other for loops where i = rows and j = cols,
        in here, i will iterate through each column once, and j
        will iterate through each record once per column to find
        the min and max of each column, representing a feature*/
        double min_so_far = dataset[0][i];
        double max_so_far = dataset[0][i];
        for(int j = 0; j < dataset.size(); j++){
            double current = dataset[j][i];
            if(current < min_so_far)
                min_so_far = current;
            else if(current > max_so_far)
                max_so_far = current;
        }
        // Set the min of column i
        min_and_max[i][0] = min_so_far;
        // Set the max of column i
        min_and_max[i][1] = max_so_far;
    }
    for(int i = 0; i < dataset[0].size(); i++){
        double min = min_and_max[i][0];
        double max = min_and_max[i][1];
        for(int j = 0; j < dataset.size(); j++){
            double current = dataset[j][i];
            normalized_dataset[j][i] = (current - min) / (max - min);
        }
    }
    return normalized_dataset;
}

double k_means(std::vector<std::vector<double>> dataset, int num_centroids){
    // Variable explanations:
    // dataset: record data indexed from 0 to dataset.size()
    // centroids: array of records that are chosen to represent the centroids, numbered from 0 to num_centroids
    // centroid_indexes: mapping of integers that represent where each record in 'centroids' can be found in the original dataset. i.e centroids[i] = dataset[centroid_indexes[i]]
    // cluster_assignments: mapping of integers that correspond with record data such that centroids[cluster_assignments[i]] == dataset[i]
    // assigned_samples: mapping

    std::vector<std::vector<double>> centroids(num_centroids, std::vector<double>());
    int centroid_indexes[num_centroids];
    srand(time(NULL));

    // Initialize centroids:
    for(int i = 0; i < num_centroids; i++){
        /* You want each centroid to be unique, so the
        do while loop below exists to keep generating a
        random number from 0 to dataset.size() - 1 until
        you have chosen the index of a sample to act as a
        centroid that you have not made before. */
        bool duplicate_centroids = false;
        do{
            int rand_centroid_index = rand() % dataset.size();
            centroids[i] = dataset[rand_centroid_index];
            centroid_indexes[i] = rand_centroid_index;
            for(int j = 0; j < num_centroids; j++)
                if(i != j && centroid_indexes[i] == centroid_indexes[j])
                   duplicate_centroids = true;
        }while(duplicate_centroids);
    }
    // SSE = sum of squared error
    double curr_SSE, last_SSE = -99, delta_SSE;
    const double THRESHOLD = 0.001;
    const int MAX_ITERATIONS = 100;
    int iterations = 0;
    std::vector<int> cluster_assignments(dataset.size(), -1);
    do{
       // First, assign all points to their centroids
        for(int i = 0; i < dataset.size(); i++){
            double closest_centroid_index = 0;
            std::vector<double> closest_centroid = centroids[closest_centroid_index];
            double closest_centroid_dist = euclid_dist(dataset[0], centroids[0]);

            // You already calculated the first one, so this loop can start at 1 instead of 0
            for(int j = 1; j < num_centroids; j++){
                double current_centroid_dist = euclid_dist(dataset[i], centroids[j]);
                if(current_centroid_dist < closest_centroid_dist){
                    closest_centroid_index = j;
                    closest_centroid_dist = current_centroid_dist;
                }
            }
            cluster_assignments[i] = closest_centroid_index;
        }
        curr_SSE = find_SSE(dataset, cluster_assignments, centroids);
        if (last_SSE == -99){
            delta_SSE = curr_SSE;
        }else{
            delta_SSE = abs(curr_SSE - last_SSE);
        }
        last_SSE = curr_SSE;
        update_centroids(dataset, cluster_assignments, centroids, num_centroids);
        iterations++;
    }while(delta_SSE > THRESHOLD && iterations < MAX_ITERATIONS);
    return last_SSE;
}

// Helper functions for the k-means method
// void update_centroids(std::vector<std::vector<double>> dataset, std::vector<int> cluster_assignments, int centroid_indexes[], std::vector<std::vector<double>>& centroids, int num_centroids){
void update_centroids(std::vector<std::vector<double>> dataset, std::vector<int> cluster_assignment, std::vector<std::vector<double>>& centroids, int num_centroids){
    // dataset: record data indexed from 0 to number of records
    // centroids: array of the records that are chosen to represent the centroids, numbered from 0 to number of centroids
    // centroid_indexes: mapping of integers that represent where each record in 'centroids' can be found in the original dataset. i.e centroids[i] = dataset[centroid_indexes[i]]
    // cluster_assignments: mapping of integers that correspond with record data and represent what cluster a given record is assigned to such that centroids[cluster_assignments[i]] == dataset[i]

    // We will calculate the average of each column for all the samples in each cluster
    std::vector<std::vector<double>> averages(num_centroids, std::vector<double>(dataset[0].size(), -1));
    int amount_in_cluster[num_centroids] = {0};

    // First, fill 'averages' with the total sum of each column per centroid
    for(int i = 0; i < dataset[0].size(); i++){
        // Go column by column, totaling up all the values in each column and storing them in averages[i]
        for(int j = 0; j < dataset.size(); j++){
            averages[cluster_assignment[j]][i] += dataset[j][i];
            amount_in_cluster[cluster_assignment[j]]++;
        }
    }
    // After sums have been totaled up, divide total by record amount in each cluster
    for(int i = 0; i < num_centroids; i++){
        for(int j = 0; j < dataset[0].size(); j++){
            // Semantically, amount of records per cluster is best represented as an integer
            // but it must be cast to a double here to avoid truncating due to integer division.
            averages[i][j] /= static_cast<double>(amount_in_cluster[i]);
        }
    }
    // After averages of each column per centroid has been found, find which record is closest to each average.
    for(int i = 0; i < num_centroids; i++){
        double last_dist = euclid_dist(dataset[0], averages[i]);
        double closest_dist = last_dist;
        for(int j = 1; j < dataset.size(); j++){
            double curr_dist = euclid_dist(dataset[j], averages[i]);
            if(curr_dist < last_dist){
                closest_dist = curr_dist;
                centroids[i] = dataset[j];
            }
        }
    }
}

double find_SSE(std::vector<std::vector<double>> dataset, std::vector<int> cluster_assignments, std::vector<std::vector<double>> centroids){
    // dataset: record data indexed from 0 to number of records
    // centroids: array of the records that are chosen to represent the centroids, numbered from 0 to number of centroids
    // centroid_indexes: mapping of integers that represent where each record in 'centroids' can be found in the original dataset. i.e centroids[i] = dataset[centroid_indexes[i]]
    // cluster_assignments: mapping of integers that correspond with record data and represent what cluster a given record is assigned to such that centroids[cluster_assignments[i]] == dataset[i]
    double SSE = 0;
    for(int i = 0; i < dataset.size(); i++){
        SSE += pow(euclid_dist(dataset[i], centroids[cluster_assignments[i]]), 2);

/* Debugging print statements below. Use these if SSE values being returned are abnormal.
        std::cout << "euclid dist between p and q, where p is: \n";
        for(int j = 0; j < dataset[0].size(); j++){
            std::cout << dataset[i][j] << " ";
        }
        std::cout << "\nand q is\n";
        for(int j = 0; j < dataset[0].size(); j++){
            std::cout << centroids[cluster_assignments[i]][j] << " ";
        }
        std::cout << " is " << euclid_dist(dataset[i], centroids[cluster_assignments[i]]) << std::endl;
*/
    }
    return SSE;
}

int main(){
    // Change these constants depending on the dataset
    const int RECORDS = 200;
    const int DIMENSIONS = 5;
    std::fstream file;
    std::string line;
    std::string curr_token;
    std::vector<std::string> columns (DIMENSIONS, "NaN");

    // One of our dimensions in the dataset 'Mall_Customers.csv'
    // is a customer ID column, so we will not include
    // it in our calculations, hence DIMENSIONS - 1
    std::vector<std::vector<double>> data (RECORDS, std::vector<double>(DIMENSIONS - 1, -1));
    file.open("Mall_Customers.csv", std::fstream::in);

    getline(file, line);
    std::stringstream ss(line);
    // Read in the first line, which is the header row and
    // does not include data
    for(int i = 0; i < DIMENSIONS; i++){
        getline(ss, curr_token, ',');
        columns[i] = curr_token;
        std::cout << "Column " << i + 1 << " is :" << columns[i] << std::endl;
    }

    for(int i = 0; i < RECORDS; i++){
        getline(file, line);
        ss = std::stringstream(line);
        int first_col = 0;
        for(int j = 0; j < DIMENSIONS; j++){
            getline(ss, curr_token, ',');
            if(first_col++ == 0){
                // Do not count first column (Customer ID).
                // Has no meaning in data besides identification.
                // Set back j by one so it still starts at index 0.
                j--;
                continue;
            }
            if(curr_token == "Male"){
                // 0 will represent male
                data[i][j] = 0;
            }else if(curr_token == "Female"){
                // 1 will represent female
                data[i][j] = 1;
            }else{
                data[i][j] = stoi(curr_token);
            }
        }
    }
    int num_centroids = 3;
    double SSE = k_means(minmax_normalize(data), num_centroids);
    std::cout << "The SSE (Sum of squared errors) running k-means on this dataset with\n";
    std::cout << num_centroids << " centroids: " << SSE << std::endl;
    num_centroids = 5;
    SSE = k_means(minmax_normalize(data), num_centroids);
    std::cout << num_centroids << " centroids: " << SSE << std::endl;
    num_centroids = 10;
    SSE = k_means(minmax_normalize(data), num_centroids);
    std::cout << num_centroids << " centroids: " << SSE << std::endl;

// Uncomment the blocks below to print out the contents of 'data'
/*    std::cout << "Non-normalized data: \n";
    for(int i = 0; i < RECORDS; i++){
        for(int j = 0; j < DIMENSIONS; j++)
            std::cout << data[i][j] << ", ";
        std::cout << std::endl;
    }
    std::vector<std::vector<double>> normalized_data = minmax_normalize(data);

    std::cout << "Normalized data: \n";
    for(int i = 0; i < RECORDS; i++){
        for(int j = 0; j < DIMENSIONS; j++)
            std::cout << normalized_data[i][j] << ", ";
        std::cout << std::endl;
    }
*/

    file.close();
    std::cout << "end of program\n";
    return 0;
}
