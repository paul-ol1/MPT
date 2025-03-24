#include <iostream>
#include <random>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>


/* This code snippet is used to compute the Global Minimum Variance Portfolio (GMVP) for a given set of assets. The GMVP is the portfolio with the lowest risk (volatility) and is calculated using the assets' historical returns and covariance matrix. The GMVP weights are calculated using the formula: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1). The expected return and volatility of the GMVP are then calculated. The code reads the CSV files saved in the data folder and computes the GMVP for different asset sizes*/
//to compile: g++ -std=c++17 compute.cpp -o compute
//to run: ./compute
// please run data_collection.py to generate the data before running this code


namespace fs = std::filesystem;
std::pair<std::vector<std::vector<float>>, std::vector<float>> read_csv(const std::string& folder_path, int amount) {
    std::vector<std::vector<float>> data;
    std::vector<std::string> csv_files;
    std::vector<float> means;

    // Get list of CSV files
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".csv") {
            csv_files.push_back(entry.path().string());
        }
    }

    //sort the files
    std::sort(csv_files.begin(), csv_files.end());
    // Limit the number of files processed based on the amount argument
    int num_files_to_process = std::min(amount, static_cast<int>(csv_files.size()));

    for (int i = 0; i < num_files_to_process; ++i) {
        std::ifstream file(csv_files[i]);  // Open only the first "amount" files
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
            }
        }

        // Calculate mean
        float mean = sum / column_data.size();
        means.push_back(mean);
        data.push_back(column_data);
    }

    return std::make_pair(data, means);
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
// Function to calculate correlation matrix
std::vector<std::vector<float> >  correlation_matrix (std::vector<std::vector<float>> data, std::vector<float> means, int M, int N) {
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
// Function to calculate Sharpe Ratio
float sharpe_ratio(std::vector<float> weights, std::vector<float> means, std::vector<std::vector<float>> covariance_matrix) {
    float sum = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        sum += weights[i] * means[i];
    }

    float variance = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights.size(); j++) {
            variance += weights[i] * weights[j] * covariance_matrix[i][j];
        }
    }

    return sum / variance;
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

std::vector<float> allOperations(std::vector<std::vector<float>> data, std::vector<float> means, int M, int N){
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> covariance_matrix = covariance(data, means, M, N);
    std::vector<std::vector<float>> coVarInv = invert_matrix(covariance_matrix);
    std::vector<float> weights = calculate_weights(coVarInv, N);
    std::vector<float> normalized_weights = normalize_weights(weights);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken to compute weights: " << duration.count() << " ms\n";
    return normalized_weights;
}

int main() {
    std::string folder_path = "/Users/pauloladele/MPT/data"; // Change this to the path of the folder containing the CSV files

    // test for different asset sizes
    int asset_size[5] ={10,100,200,500,955};
    for(int i =0;i<5;i++){
    auto data_means = read_csv(folder_path,asset_size[i]);
    std::vector<std::vector<float>> data = data_means.first;
    std::vector<float> means = data_means.second;


    int N = data.size();
    int M = data[0].size();
    std::cout<< "N: " << N << " M: " << M << std::endl;
    std::vector<float> weights = allOperations(data, means, M, N);
    std::cout << "Weights: ";
    for (int i = 0; i < N; i++){
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;

    }

    return 0;
}
