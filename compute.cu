#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>


/* This code snippet is used to compute the Global Minimum Variance Portfolio (GMVP) for a given set of assets. The GMVP is the portfolio with the lowest risk (volatility) and is calculated using the assets' historical returns and covariance matrix. The GMVP weights are calculated using the formula: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1). The expected return and volatility of the GMVP are then calculated. The code reads the CSV files saved in the data folder and computes the GMVP for different asset sizes*/

// to compile:: nvcc -o compute compute.cu -std=c++17 -lcublas -lcurand -lcusolver -lstdc++fs
// to run:: ./compute
//run data_collection.py to get the data first before running this code

// this code is used to compute the weights of the GMV portfolio
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

    // Limit the number of files processed based on the "amount" argument
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




__global__ void covar(float *data, float *means, float *covar, int dayscount, int num_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_id = index % num_features;
    int y_id = index / num_features;
    float sum = 0.0;

    if (x_id < num_features && y_id < num_features) {
        // Compute covariance for feature pairs (x_id, y_id)
        for (int i = 0; i < dayscount; i++) {
            float diff_x = data[x_id * dayscount + i] - means[x_id];
            float diff_y = data[y_id * dayscount + i] - means[y_id];
            sum += diff_x * diff_y;
        }

        // Compute final covariance value
        covar[x_id * num_features + y_id] = sum / (dayscount - 1);


    }
}

// correlation matrix


// to get the weights a matrix vector mult is performed where each matrix row is basically added together by using vectors of 1
__global__ void GMVPotfolio(float *covInv, float *ones, float *weights, int num_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_features) {
        float sum = 0;
        for (int i = 0; i < num_features; i++) {
            sum += covInv[i * num_features + index] * ones[i];
        }
        weights[index] = sum;
    }
}

// get 1D matrices for memory
std::vector<float> flatten_matrix(std::vector<std::vector<float>> data){
    std::vector<float> flat_data;
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            flat_data.push_back(data[i][j]);
        }
    }
    return flat_data;
}
// convert row to column major(unused)
std::vector<float> rowToColumnMajor(std::vector<float> data, int numOfFeatures) {
    std::vector<float> column_major;
    for (int i = 0; i < numOfFeatures * numOfFeatures; i++) {
        int row = i / numOfFeatures;
        int col = i % numOfFeatures;
        column_major.push_back(data[row + col * numOfFeatures]);

    }
    return column_major;
}

// standard deviation of the data
std::vector<float> standard_deviations(std::vector<std::vector<float>> data){
    std::vector<float> std_devs;
    for(int i = 0; i < data.size(); i++){
        float sum = 0;
        for(int j = 0; j < data[i].size(); j++){
            sum += data[i][j];
        }
        float mean = sum / data[i].size();
        float variance = 0;
        for(int j = 0; j < data[i].size(); j++){
            variance += pow(data[i][j] - mean, 2);
        }
        float std_dev = sqrt(variance / data[i].size());
        std_devs.push_back(std_dev);
    }
    return std_devs;
}

// portfolio standard deviation
float portfolio_std_dev(std::vector<float> weights, std::vector<float> std_devs, std::vector<float> corr_matrix) {
    float variance = 0.0f;
    int N = weights.size();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float term = weights[i] * weights[j] * std_devs[i] * std_devs[j] * corr_matrix[i * N + j];
            variance += term;
        }
    }

    return sqrt(variance);
}

// portfolio return
float portfolio_return(std::vector<float> weights, std::vector<float> means) {
    float ret = 0.0f;
    for (int i = 0; i < weights.size(); i++) {
        ret += weights[i] * means[i];
    }
    return ret;
}

// correlation matrix calculation for the portfolio based computations
std::vector<float> corr_matrix(std::vector<float>covariance_matrix, std::vector<float> std_devs){
    std::vector<float> correlation_matrix;
    for(int i = 0; i < std_devs.size(); i++){
        for(int j = 0; j < std_devs.size(); j++){
            float correlation = covariance_matrix[i * std_devs.size() + j] / (std_devs[i] * std_devs[j]);
            correlation_matrix.push_back(correlation);
        }}
    return correlation_matrix;
}

// normalize the weights by dividing by the sum of the weights
std::vector<float> normalize_weights(std::vector<float> weights){
    float sum = 0;
    for(int i = 0; i < weights.size(); i++){
        sum += weights[i];
    }
    for (int i = 0; i < weights.size(); i++) {
        weights[i] /= sum;
    }
    return weights;
}

// kernel operations for all the computations and memory management on the GPU
std::vector<float> kernelOperations(std::vector<float> data, std::vector<float> means, int M, int N, int threadsize, std::vector<float> std_devs) {
    int blocksize = (N * N + threadsize - 1) / threadsize; // Number of blocks needed

    // Memory allocation
    float matrix_size = M * N * sizeof(float);
    std::vector<float> h_covar(N * N);
    std::vector<float> weights(N, 0.0f);
    float *d_data, *d_covar, *d_means, *d_weights, *d_ones,*d_identity, *d_work;
    int *gpu_pivot, *gpu_info;
    int info_gpu = 0;
    cublasHandle_t handle;
    cusolverDnHandle_t cuhandle;

    cudaMalloc(&d_data, matrix_size);
    cudaMalloc(&d_covar, N * N * sizeof(float));
    cudaMalloc(&d_means, N * sizeof(float));
    cudaMalloc(&d_identity, N * N * sizeof(float));
    cudaMalloc(&d_work, N * N * sizeof(float));
    cudaMalloc(&gpu_pivot, N * sizeof(int));
    cudaMalloc(&gpu_info, sizeof(int));
    cudaMalloc(&d_ones, N * sizeof(float));
    cudaMalloc(&d_weights, N * sizeof(float));

    std::vector<float> ones(N, 1.0f);  // Initialize ones vector
    cudaMemcpy(d_ones, ones.data(), N * sizeof(float), cudaMemcpyHostToDevice);


    // cuBLAS/cuSOLVER handles
    cublasCreate(&handle);
    cusolverDnCreate(&cuhandle);

    // Copy inputs
    cudaMemcpy(d_data, data.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    // Launch covariance kernel & calculate covariance matrix on GPU
    covar<<<blocksize, threadsize>>>(d_data, d_means, d_covar, M, N);
    cudaDeviceSynchronize();


    cudaMemcpy(h_covar.data(), d_covar, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    In order to calculate the GMV portfolio weights, we need to calculate the inverse of the covariance matrix. We use the cuSOLVER library to perform LU decomposition and matrix inversion. The LU decomposition is done in-place on the covariance matrix, and the inverse is calculated using the LU factors.
    */

    // LU decomposition of covariance matrix
    cusolverDnSgetrf(cuhandle, N, N, d_covar, N, d_work, gpu_pivot, gpu_info);
    cudaMemcpy(&info_gpu, gpu_info, sizeof(int), cudaMemcpyDeviceToHost); // check for errors
    if (info_gpu != 0) {
        std::cerr << "LU decomposition failed! info = " << info_gpu << std::endl;
        return {};
    }

    std::vector<float> h_identity(N * N, 0.0f); // Initialize identity matrix
    for (int i = 0; i < N; ++i) {
        h_identity[i * N + i] = 1.0f;
    }
    cudaMemcpy(d_identity, h_identity.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Solve A * X = I -> X = A^{-1} using LU factors from getrf  and the identity matrix
    cusolverDnSgetrs( cuhandle,CUBLAS_OP_N,N,N,d_covar,N,gpu_pivot,d_identity,N,gpu_info);
    cudaMemcpy(&info_gpu, gpu_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_gpu != 0) {
        std::cerr << "Matrix inversion failed in getrs! info = " << info_gpu << std::endl;
        return {};
    }

    // calculate the GMV portfolio weights unNormalized GEMV operation
    GMVPotfolio<<<(N + threadsize - 1) / threadsize, threadsize>>>(d_identity, d_ones, d_weights, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);//time taken to compute the weights and just kernel operations
    cudaMemcpy(weights.data(), d_weights, N * sizeof(float), cudaMemcpyDeviceToHost);

    //testing
    weights = normalize_weights(weights);
    std::vector<float> correlation_matrix = corr_matrix(h_covar, std_devs);
    float std_dev = portfolio_std_dev(weights, std_devs, correlation_matrix);
    float ret = portfolio_return(weights, means);
    std::cout << "Portfolio Return: " << ret << "\n";
    std::cout << "Portfolio Standard Deviation: " << std_dev << "\n";
    std::cout<<"sharpe ratio: "<<ret/std_dev<<std::endl;
    std::cout<<"Time taken to compute weights: "<<duration.count()<<" ms\n";
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_covar);
    cudaFree(d_means);
    cudaFree(d_identity);
    cudaFree(d_work);
    cudaFree(gpu_pivot);
    cudaFree(gpu_info);
    cublasDestroy(handle);
    cusolverDnDestroy(cuhandle);

    return weights;
}

// basically used to structure how the operations are done and the data is passed to the kernel
std::vector<float> getWeights(std::vector<std::vector<float>> data, std::vector<float> means, int M, int N, int threadsize){
    std::vector<float> flat_data = flatten_matrix(data);
    std::vector<float> weights = kernelOperations(flat_data, means, M, N, threadsize, standard_deviations(data));
    weights = normalize_weights(weights);
    return weights;
}




int main() {
    std::string folder_path = "/data/csslab-parsons/paulol/MPT/data"; // Path to the data folder, please change this to your own path and make sure the data is in the correct format

    // Asset sizes and block sizes to ensure appropriate memory allocation
    int asset_size[5] ={10,100,200,500,955};
    int blocksizes[5] = {32,256,512,512,512};

    // Loop through the different asset sizes
    for(int i = 0; i < sizeof(asset_size)/sizeof(asset_size[0]); i++ ){
        std::pair<std::vector<std::vector<float>>, std::vector<float>> data_means = read_csv(folder_path, asset_size[i]);
        std::vector<std::vector<float>> data = data_means.first;
        std::vector<float> means = data_means.second;
        std::cout << "\nData size: " << data.size() << " x " << data[0].size() << "\n";
        std::vector<float> flat_data = flatten_matrix(data);
        std::vector<float> std_devs = standard_deviations(data);
        std::vector<float> weights = kernelOperations(flat_data, means, data[0].size(),data.size(), blocksizes[i],std_devs);

        std::cout << "\n";
        std::cout << "Weights:[ ";
        for(int i = 0; i < weights.size(); i++){
            std::cout << weights[i] << ", ";
        }
        std::cout << "]\n";


    }
    return 0;
}