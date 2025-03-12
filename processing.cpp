#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <sstream>


// to compile: g++ -std=c++17 -o processing processing.cpp
namespace fs = std::filesystem;

std::vector<std::vector<double>> read_csv(const std::string& folder_path) {
    std::vector<std::vector<double>> data;
    std::vector<std::string> csv_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".csv") {
            csv_files.push_back(entry.path().string());
        }
    }

    // Sort file paths alphabetically
    std::sort(csv_files.begin(), csv_files.end());

    for (const auto& entry : csv_files) {

            std::ifstream file(entry);
            std::vector<double> column_data;
            std::string line;

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                int column_index = 0;

                while (std::getline(ss, cell, ',')) {
                    if (column_index == 1) { // Second column (zero-based index)
                        try {
                            column_data.push_back(std::stod(cell)); // Convert to double
                        } catch (const std::exception& e) {
                            std::cerr << "Skipping invalid data: " << cell << " in file " << entry<< "\n";
                        }
                        break; // Stop reading after second column
                    }
                    column_index++;
                }
            }
            data.push_back(column_data);

    }
    return data;
}

int main() {
    std::string folder_path = "/Users/pauloladele/MPT/data"; // Change this to your folder path
    auto csv_columns = read_csv(folder_path);

    // Print results
    for (int i = 0; i < csv_columns.size(); ++i) {
        std::cout << csv_columns[i].size() << " elements in column " << i << ":\n";
        if(i ==0){
            std::cout << "The first column is: \n";
            for (int j = 0; j < csv_columns[i].size(); ++j) {
                std::cout << csv_columns[i][j] << "\n";
            }
        }
    }

    return 0;
}
