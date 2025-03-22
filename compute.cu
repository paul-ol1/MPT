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



// to compile:: nvcc -o compute compute.cu -std=c++17 -lcublas -lcurand -lcusolver -lstdc++fs
// to run:: ./compute
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
                std::cerr << "Skipping invalid data: " << line << " in file " << entry << "\n";
            }

        }

        // Calculate mean
        float mean = sum / column_data.size();
        means.push_back(mean);
        data.push_back(column_data);
    }

    return  std::make_pair(data, means);
}



__global__ void covar(float *data, float *means, float *covar, int weekscount, int num_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_id = index % num_features;  
    int y_id = index / num_features;
    float sum = 0.0;

    if (x_id < num_features && y_id < num_features) {
        // Compute covariance for feature pairs (x_id, y_id)
        for (int i = 0; i < weekscount; i++) {
            float diff_x = data[x_id * weekscount + i] - means[x_id];
            float diff_y = data[y_id * weekscount + i] - means[y_id];
            sum += diff_x * diff_y;
        }

        // Compute final covariance value
        covar[x_id * num_features + y_id] = sum / (weekscount - 1);


    }
}

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
std::vector<float> rowToColumnMajor(std::vector<float> data, int numOfFeatures) {
    std::vector<float> column_major;
    for (int i = 0; i < numOfFeatures * numOfFeatures; i++) {
        int row = i / numOfFeatures;
        int col = i % numOfFeatures;
        column_major.push_back(data[row + col * numOfFeatures]);

    }
    return column_major;
}


std::vector<float> kernelOperations(std::vector<float> data, std::vector<float> means, int M, int N, int blocksize, int threadsize) {
    float matrix_size = M * N * sizeof(float);
    float h_covar[N * N];
    std::vector<float> weights(N, 0.0f);
    float *d_data, *d_covar, *d_means, *d_weights, *d_ones,*d_identity, *d_work;
    int *gpu_pivot, *gpu_info;
    int info_gpu = 0;
    cublasHandle_t handle;
    cusolverDnHandle_t cuhandle;

    // Allocate memory
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

    // Launch covariance kernel & calculate covariance matrix on GPU
    covar<<<blocksize, threadsize>>>(d_data, d_means, d_covar, M, N);
    cudaDeviceSynchronize();


    cudaMemcpy(h_covar, d_covar, N * N * sizeof(float), cudaMemcpyDeviceToHost);

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

    // calculate the GMV portfolio weights unNormalized
    if(N/ threadsize > 0){
        GMVPotfolio<<<N/threadsize, threadsize>>>(d_identity, d_ones, d_weights, N);
    }else{
        GMVPotfolio<<<1, N>>>(d_identity, d_ones, d_weights, N);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(weights.data(), d_weights, N * sizeof(float), cudaMemcpyDeviceToHost);


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




int main() {
    std::string folder_path = "/data/csslab-parsons/paulol/MPT/data";
    std::pair<std::vector<std::vector<float>>, std::vector<float>> data_means = read_csv(folder_path);
    std::vector<std::vector<float>> data = data_means.first;
    std::vector<float> means = data_means.second;
    std::cout << "\n";
    std::cout << "Data size: " << data.size() << " x " << data[0].size() << "\n";
    std::vector<float> flat_data = flatten_matrix(data);
    std::vector<float> weights = kernelOperations(flat_data, means, data[0].size(),data.size(), 471, 512);
    float sum = 0;
    for(int i = 0; i < weights.size(); i++){
        sum += weights[i];
    }
    // Normalize weights
    for (int i = 0; i < weights.size(); i++) {
        weights[i] /= sum;
    }
    for(int i = 0; i < weights.size(); i++){
        std::cout << weights[i] << " ";
    }


    return 0;
}