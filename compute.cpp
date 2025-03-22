#include <iostream>
#include <random>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>


//to compile: g++ -std=c++17 compute.cpp -o compute
//to run: ./compute


namespace fs = std::filesystem;

std::pair<std::vector<std::vector<float>>,std::vector<float>>  read_csv(const std::string& folder_path) {
    std::vector<std::vector<float>> data;
    std::vector<std::string> csv_files;
    std::vector<float> means;

    // Get list of CSV files
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".csv") {
            csv_files.push_back(entry.path().string());
        }
    }
 // Sort the csv_files vector in alphabetical order
    std::sort(csv_files.begin(), csv_files.end());

    for (const auto& entry : csv_files) {
        std::ifstream file(entry);
        std::vector<float> column_data;
        std::string line;

        bool first_row = true;
        float prev_price = 0.0;  // Store previous day's price
        float sum = 0.0;  // Store sum of prices for mean calculation

        while (std::getline(file, line)) {
            std::stringstream ss(line);



            try {
                float price = std::stod(line);
                if (first_row) {
                    prev_price = price;  // Store first price but don't calculate return
                    first_row = false;
                } else {
                    float log_return = log(price / prev_price);
                    column_data.push_back(log_return);
                    sum += log_return;
                    prev_price = price;  // Update for next iteration
                }
            } catch (const std::exception& e) {
                std::cerr << "Completing: " << line  << entry << "\n";
            }

        }

        // Calculate mean
        float mean = sum / column_data.size();
        means.push_back(mean);
        data.push_back(column_data);
    }

    return  std::make_pair(data, means);
}


// Function to calculate covariance matrix
std::vector<std::vector<float> >  covariance (std::vector<std::vector<float>> data, std::vector<float> means, int M, int N) {
    std::vector<std::vector<float>> covariance_matrix(N, std::vector<float>(N, 0.0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < M; k++){
                covariance_matrix[i][j] += (data[i][k] - means[i]) * (data[j][k] - means[j]);
            }
            covariance_matrix[i][j] /= M-1;
    }}

    return covariance_matrix;
}
// invert covariance matrix
std::vector<std::vector<float>> invert_matrix(std::vector<std::vector<float>> matrix) {
    int n = matrix.size();
    std::vector<std::vector<float>> inverse(n, std::vector<float>(n, 0.0));

    // Initialize inverse as identity matrix
    for (int i = 0; i < n; ++i)
        inverse[i][i] = 1.0;

    for (int i = 0; i < n; ++i) {
        float pivot = matrix[i][i];
        if (pivot == 0.0) {
            std::cerr << "Matrix is singular and cannot be inverted.\n";
            exit(1);
        }

        // Normalize pivot row
        for (int j = 0; j < n; ++j) {
            matrix[i][j] /= pivot;
            inverse[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            float factor = matrix[k][i];
            for (int j = 0; j < n; ++j) {
                matrix[k][j] -= factor * matrix[i][j];
                inverse[k][j] -= factor * inverse[i][j];
            }
        }
    }

    return inverse;
}

// calculate the weights
std::vector<float> calculate_weights(std::vector<std::vector<float>> coVarInv, int N){

    std::vector<float> weights(N, 0.0);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            weights[i] += coVarInv[i][j];
        }
    }
    return weights;
}
// normalize the weights
std::vector<float> normalize_weights(std::vector<float> weights){
    int n = weights.size();
    float sum = 0.0;
    for (int i = 0; i < n; ++i){
        sum += weights[i];
    }
    for (int i = 0; i < n; ++i){
        weights[i] /= sum;
    }
    return weights;
}

int main() {
    std::string folder_path = "/Users/pauloladele/MPT/data";
    auto data_means = read_csv(folder_path);
    std::vector<std::vector<float>> data = data_means.first;
    std::vector<float> means = data_means.second;


    int N = data.size();
    int M = data[0].size();
    // Compute covariance matrix
    std::vector<std::vector<float>> covariance_matrix = covariance(data, means, M, N);

    // Compute inverse covariance matrix
    std::vector<std::vector<float>> coVarInv = invert_matrix(covariance_matrix);


    // Compute and normalize weights
    std::vector<float> weights = calculate_weights(coVarInv, N);
    std::vector<float> normalized_weights = normalize_weights(weights);
    printf("Weights: ");
    for (int i = 0; i < N; i++){
        std::cout << normalized_weights[i] << " ";
    }
    printf("\n");


    return 0;
}
