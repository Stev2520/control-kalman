// analysis_tools.hpp
#ifndef ANALYSIS_TOOLS_HPP
#define ANALYSIS_TOOLS_HPP

#include "data_generator.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace analysis {

    struct FilterPerformance {
        double average_error = 0.0; // Общая ошибка
        double max_error = 0.0; // Максимальная ошибка
        double rmse = 0.0;  // Root Mean Square Error
        double convergence_time = 0.0;  // Время сходимости
        std::vector<double> error_history; // Вектор ошибки
    };

    FilterPerformance analyze_filter(const std::vector<Eigen::Vector2d>& true_states,
                                     const std::vector<Eigen::Vector2d>& estimates,
                                     const std::vector<double>& times) {
        FilterPerformance perf;
        double sum_sq_error = 0.0;
        perf.max_error = 0.0;
        perf.error_history.reserve(true_states.size());

        for (size_t i = 0; i < true_states.size(); ++i) {
            double error = (true_states[i] - estimates[i]).norm();
            perf.error_history.push_back(error);
            sum_sq_error += error * error;
            if (error > perf.max_error) {
                perf.max_error = error;
            }
            // Определяем время сходимости
            if (i > 100 && error < 0.01) {  // Примерный критерий
                perf.convergence_time = times[i];
            }
        }

        perf.average_error = std::accumulate(perf.error_history.begin(),
                                             perf.error_history.end(), 0.0)
                             / perf.error_history.size();

        perf.rmse = std::sqrt(sum_sq_error / perf.error_history.size());
        return perf;
    }

    void generate_comparison_report(const data_generator::SimulationData& data,
                                    const std::string& output_file)
    {
        auto ckf_perf = analyze_filter(data.true_states,
                                       data.ckf_estimates,
                                       data.times);
        auto srcf_perf = analyze_filter(data.true_states,
                                        data.srcf_estimates,
                                        data.times);

        std::ofstream report(output_file);

        report << "=== Kalman Filter Performance Comparison ===\n\n";

        report << "CKF Performance:\n";
        report << "  Average error: " << ckf_perf.average_error << "\n";
        report << "  Max error: " << ckf_perf.max_error << "\n";
        report << "  RMSE: " << ckf_perf.rmse << "\n";
        report << "  Convergence time: " << ckf_perf.convergence_time << " s\n\n";

        report << "SRCF Performance:\n";
        report << "  Average error: " << srcf_perf.average_error << "\n";
        report << "  Max error: " << srcf_perf.max_error << "\n";
        report << "  RMSE: " << srcf_perf.rmse << "\n";
        report << "  Convergence time: " << srcf_perf.convergence_time << " s\n\n";

        report << "Comparison:\n";
        report << "  SRCF/CKF error ratio: " << srcf_perf.average_error / ckf_perf.average_error << "\n";
        report << "  SRCF/CKF RMSE ratio: " << srcf_perf.rmse / ckf_perf.rmse << "\n";

        if (srcf_perf.average_error > ckf_perf.average_error * 1.2) {
            report << "\nCONCLUSION: CKF performs better than SRCF\n";
        } else if (srcf_perf.average_error < ckf_perf.average_error) {
            report << "\nCONCLUSION: SRCF performs better than CKF\n";
        } else {
            report << "\nCONCLUSION: Both filters have similar performance\n";
        }

        report.close();
    }

} // namespace analysis

#endif // ANALYSIS_TOOLS_HPP