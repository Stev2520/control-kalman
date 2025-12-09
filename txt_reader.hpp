// txt_reader.hpp
#ifndef TXT_READER_HPP
#define TXT_READER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>

namespace txt_reader {

    struct SimulationData {
        std::vector<double> times;
        std::vector<Eigen::Vector2d> true_states;
        std::vector<Eigen::Vector2d> ckf_estimates;
        std::vector<Eigen::Vector2d> srcf_estimates;
        std::vector<double> controls;
    };

    SimulationData readMainData(const std::string& filename) {
        SimulationData data;
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;

        // Пропускаем заголовки
        for (int i = 0; i < 4; ++i) {
            std::getline(file, line);
        }

        // Читаем данные
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '=' || line[0] == '-') {
                continue;
            }

            std::istringstream iss(line);
            double time, true_phi, true_p, ckf_phi, ckf_p, srcf_phi, srcf_p, control;

            if (iss >> time >> true_phi >> true_p >> ckf_phi >> ckf_p >> srcf_phi >> srcf_p >> control) {
                data.times.push_back(time);
                data.true_states.push_back(Eigen::Vector2d(true_phi, true_p));
                data.ckf_estimates.push_back(Eigen::Vector2d(ckf_phi, ckf_p));
                data.srcf_estimates.push_back(Eigen::Vector2d(srcf_phi, srcf_p));
                data.controls.push_back(control);
            }
        }

        file.close();
        return data;
    }

    void printDataSummary(const SimulationData& data) {
        std::cout << "\n=== Data Summary ===\n";
        std::cout << "Number of samples: " << data.times.size() << "\n";

        if (!data.times.empty()) {
            std::cout << "Time range: " << data.times.front() << " to "
                      << data.times.back() << " seconds\n";
            std::cout << "Duration: " << data.times.back() - data.times.front()
                      << " seconds\n";
        }

        if (!data.true_states.empty()) {
            Eigen::Vector2d avg_state = Eigen::Vector2d::Zero();
            for (const auto& state : data.true_states) {
                avg_state += state;
            }
            avg_state /= data.true_states.size();
            std::cout << "Average true state: [" << avg_state(0) << ", "
                      << avg_state(1) << "]\n";
        }
    }

} // namespace txt_reader

#endif // TXT_READER_HPP