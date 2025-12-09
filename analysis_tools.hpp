// analysis_tools.hpp
#ifndef ANALYSIS_TOOLS_HPP
#define ANALYSIS_TOOLS_HPP

#include "data_generator.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace analysis {
    // Структура для хранения метрик
    struct FilterMetrics {
        double average_error = 0.0; // Общая ошибка
        double max_error = 0.0; // Максимальная ошибка
        double rms_error = 0.0;  // Root Mean Square Error
        double convergence_time = 0.0;  // Время сходимости
        double cov_norm = 0.0;      // норма ковариации
        double cond_number = 0.0;   // число обусловленности
        double symmetry_error = 0.0; // асимметрия ковариации
        std::vector<double> error_history; // Вектор ошибки
    };

    FilterMetrics analyze_filter(const std::vector<Eigen::Vector2d>& true_states,
                                     const std::vector<Eigen::Vector2d>& estimates,
                                     const std::vector<double>& times) {
        FilterMetrics perf;
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

        perf.rms_error = std::sqrt(sum_sq_error / perf.error_history.size());
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
        report << "  RMSE: " << ckf_perf.rms_error << "\n";
        report << "  Convergence time: " << ckf_perf.convergence_time << " s\n\n";

        report << "SRCF Performance:\n";
        report << "  Average error: " << srcf_perf.average_error << "\n";
        report << "  Max error: " << srcf_perf.max_error << "\n";
        report << "  RMSE: " << srcf_perf.rms_error << "\n";
        report << "  Convergence time: " << srcf_perf.convergence_time << " s\n\n";

        report << "Comparison:\n";
        report << "  SRCF/CKF error ratio: " << srcf_perf.average_error / ckf_perf.average_error << "\n";
        report << "  SRCF/CKF RMSE ratio: " << srcf_perf.rms_error / ckf_perf.rms_error << "\n";

        if (srcf_perf.average_error > ckf_perf.average_error * 1.2) {
            report << "\nCONCLUSION: CKF performs better than SRCF\n";
        } else if (srcf_perf.average_error < ckf_perf.average_error) {
            report << "\nCONCLUSION: SRCF performs better than CKF\n";
        } else {
            report << "\nCONCLUSION: Both filters have similar performance\n";
        }

        report.close();
    }

    void analyze_verhaegen_style(const data_generator::SimulationData& data) {
        std::cout << "\n=== ANALYSIS IN VERHAEGEN & VAN DOOREN STYLE ===\n";

        // 1. Анализ численной устойчивости
        double max_ckf_asymmetry = 0.0;
        double max_srcf_asymmetry = 0.0;
        int ckf_non_positive = 0;
        int srcf_non_positive = 0;

        for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
            // Асимметрия: ||P - P^T|| / ||P||
            Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] - data.ckf_covariances[i].transpose();
            Eigen::Matrix2d asym_srcf = data.srcf_covariances[i] - data.srcf_covariances[i].transpose();

            double asym_ckf_norm = asym_ckf.norm() / data.ckf_covariances[i].norm();
            double asym_srcf_norm = asym_srcf.norm() / data.srcf_covariances[i].norm();

            max_ckf_asymmetry = std::max(max_ckf_asymmetry, asym_ckf_norm);
            max_srcf_asymmetry = std::max(max_srcf_asymmetry, asym_srcf_norm);

            // Положительная определенность
            Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
            Eigen::LLT<Eigen::Matrix2d> llt_srcf(data.srcf_covariances[i]);

            if (llt_ckf.info() != Eigen::Success) ckf_non_positive++;
            if (llt_srcf.info() != Eigen::Success) srcf_non_positive++;
        }

        std::cout << "Symmetry Analysis:\n";
        std::cout << "  CKF maximum asymmetry: " << max_ckf_asymmetry << "\n";
        std::cout << "  SRCF maximum asymmetry: " << max_srcf_asymmetry << "\n";

        std::cout << "\nPositive Definiteness:\n";
        std::cout << "  CKF non-positive definite: " << ckf_non_positive
                  << "/" << data.ckf_covariances.size() << "\n";
        std::cout << "  SRCF non-positive definite: " << srcf_non_positive
                  << "/" << data.srcf_covariances.size() << "\n";

        // 2. Анализ ошибок фильтрации
        std::vector<double> ckf_errors;
        std::vector<double> srcf_errors;
        std::vector<double> innovation_norms;

        for (size_t i = 0; i < data.true_states.size(); ++i) {
            double ckf_err = (data.true_states[i] - data.ckf_estimates[i]).norm();
            double srcf_err = (data.true_states[i] - data.srcf_estimates[i]).norm();

            ckf_errors.push_back(ckf_err);
            srcf_errors.push_back(srcf_err);

            if (i > 0) {
                Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
                innovation_norms.push_back(innov.norm());
            }
        }

        // 3. Вычисление метрик
        FilterMetrics ckf_metrics, srcf_metrics;

        ckf_metrics.average_error = std::accumulate(ckf_errors.begin(), ckf_errors.end(), 0.0) / ckf_errors.size();
        srcf_metrics.average_error = std::accumulate(srcf_errors.begin(), srcf_errors.end(), 0.0) / srcf_errors.size();

        ckf_metrics.max_error = *std::max_element(ckf_errors.begin(), ckf_errors.end());
        srcf_metrics.max_error = *std::max_element(srcf_errors.begin(), srcf_errors.end());

        double ckf_sq_sum = 0.0, srcf_sq_sum = 0.0;
        for (size_t i = 0; i < ckf_errors.size(); ++i) {
            ckf_sq_sum += ckf_errors[i] * ckf_errors[i];
            srcf_sq_sum += srcf_errors[i] * srcf_errors[i];
        }
        ckf_metrics.rms_error = sqrt(ckf_sq_sum / ckf_errors.size());
        srcf_metrics.rms_error = sqrt(srcf_sq_sum / srcf_errors.size());

        double ckf_cov_norm_sum = 0.0, srcf_cov_norm_sum = 0.0;
        double ckf_cond_sum = 0.0, srcf_cond_sum = 0.0;

        for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
            ckf_cov_norm_sum += data.ckf_covariances[i].norm();
            srcf_cov_norm_sum += data.srcf_covariances[i].norm();

            Eigen::JacobiSVD<Eigen::Matrix2d> svd_ckf(data.ckf_covariances[i]);
            Eigen::JacobiSVD<Eigen::Matrix2d> svd_srcf(data.srcf_covariances[i]);

            double cond_ckf = svd_ckf.singularValues()(0) / svd_ckf.singularValues()(svd_ckf.singularValues().size()-1);
            double cond_srcf = svd_srcf.singularValues()(0) / svd_srcf.singularValues()(svd_srcf.singularValues().size()-1);

            ckf_cond_sum += cond_ckf;
            srcf_cond_sum += cond_srcf;
        }

        ckf_metrics.cov_norm = ckf_cov_norm_sum / data.ckf_covariances.size();
        srcf_metrics.cov_norm = srcf_cov_norm_sum / data.srcf_covariances.size();
        ckf_metrics.cond_number = ckf_cond_sum / data.ckf_covariances.size();
        srcf_metrics.cond_number = srcf_cond_sum / data.srcf_covariances.size();

        // 4. Вывод результатов
        std::cout << "\n=== PERFORMANCE METRICS ===\n";
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\nCKF Metrics:\n";
        std::cout << "  Average error:  " << ckf_metrics.average_error << "\n";
        std::cout << "  Maximum error:  " << ckf_metrics.max_error << "\n";
        std::cout << "  RMS error:      " << ckf_metrics.rms_error << "\n";
        std::cout << "  Avg cov norm:   " << ckf_metrics.cov_norm << "\n";
        std::cout << "  Avg cond num:   " << ckf_metrics.cond_number << "\n";

        std::cout << "\nSRCF Metrics:\n";
        std::cout << "  Average error:  " << srcf_metrics.average_error << "\n";
        std::cout << "  Maximum error:  " << srcf_metrics.max_error << "\n";
        std::cout << "  RMS error:      " << srcf_metrics.rms_error << "\n";
        std::cout << "  Avg cov norm:   " << srcf_metrics.cov_norm << "\n";
        std::cout << "  Avg cond num:   " << srcf_metrics.cond_number << "\n";

        std::cout << "\n=== COMPARATIVE ANALYSIS ===\n";
        std::cout << "SRCF/CKF error ratio:      " << srcf_metrics.average_error / ckf_metrics.average_error << "\n";
        std::cout << "SRCF/CKF RMS ratio:        " << srcf_metrics.rms_error / ckf_metrics.rms_error << "\n";
        std::cout << "SRCF/CKF condition ratio:  " << srcf_metrics.cond_number / ckf_metrics.cond_number << "\n";

        // 7. Рекомендации как в статье
        std::cout << "\n=== RECOMMENDATIONS (from Verhaegen & Van Dooren) ===\n";
        if (max_ckf_asymmetry > 1e-10) {
            std::cout << "WARNING: CKF shows significant asymmetry (" << max_ckf_asymmetry << ")\n";
            std::cout << "  Recommendation: Use Joseph stabilized form or symmetrize P at each step\n";
        }

        if (ckf_non_positive > 0) {
            std::cout << "WARNING: CKF produced non-positive definite covariance matrices\n";
            std::cout << "  Recommendation: Consider SRCF for guaranteed positive definiteness\n";
        }

        if (srcf_metrics.cond_number > 1e6) {
            std::cout << "WARNING: High condition number in SRCF (" << srcf_metrics.cond_number << ")\n";
            std::cout << "  Recommendation: Check measurement scaling or use sequential processing\n";
        }

        // Сохранение данных для графика
        std::ofstream error_file("verhaegen_analysis.csv");
        error_file << "Step,CKF_Error,SRCF_Error,CKF_Cov,SRCF_Cov,Innovation\n";

        for (size_t i = 0; i < data.true_states.size(); ++i) {
            error_file << i << ","
                       << ckf_errors[i] << ","
                       << srcf_errors[i] << ","
                       << data.ckf_covariances[i].norm() << ","
                       << data.srcf_covariances[i].norm() << ",";

            if (i > 0) {
                Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
                error_file << innov.norm();
            } else {
                error_file << "0";
            }
            error_file << "\n";
        }
        error_file.close();

        std::cout << "\nData saved to: verhaegen_analysis.csv\n";
    }

} // namespace analysis

#endif // ANALYSIS_TOOLS_HPP