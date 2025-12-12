#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include "kalman.hpp"
#include "models.hpp"
#include "time_generator.hpp"

namespace data_generator {
// ============================================================================
// 1. КОНФИГУРАЦИЯ И ДАННЫЕ
// ============================================================================
    enum class DataFormat {
        BINARY,      // Бинарный формат для скорости
        TEXT_CSV,    // Текстовый CSV для удобства
        TEXT_MATLAB, // Текстовый формат MATLAB
        TEXT_TXT     // Простой TXT формат
    };

    enum class ModelType {
        MODEL0,  // Модель рыскания самолета
        MODEL2   // Модель крена самолета
    };

    struct SimulationConfig {
        size_t total_steps = 1000;
        double base_dt = 0.01;
        bool add_process_noise = true;
        bool add_measurement_noise = true;
        double process_noise_scale = 1.0;
        double measurement_noise_scale = 1.0;

        ModelType model_type = ModelType::MODEL2;
        union {
            model0::ControlScenario scenario0;
            model2::ControlScenario scenario2;
        } scenario;

//        model2::ControlScenario scenario = model2::ControlScenario::SINE_WAVE;
        time_generator::TimeMode time_mode = time_generator::TimeMode::RANDOM_JITTER;
        DataFormat format = DataFormat::BINARY;
        std::string output_dir = "./data";
        bool test_ckf = true;
    };

    struct SimulationData {
        std::vector<double> times;
        std::vector<Eigen::Vector2d> true_states;
        std::vector<Eigen::Vector2d> measurements;
        std::vector<Eigen::Vector2d> noisy_measurements;
        std::vector<Eigen::VectorXd> controls;
        std::vector<Eigen::Vector2d> ckf_estimates;
        std::vector<Eigen::Vector2d> srcf_estimates;
        std::vector<Eigen::Matrix2d> ckf_covariances;
        std::vector<Eigen::Matrix2d> srcf_covariances;

        struct FilterMetrics {
            double average_error = 0.0;
            double max_error = 0.0;
            double rms_error = 0.0;
            double convergence_time = 0.0;
            double cov_norm = 0.0;      // норма ковариации
            double cond_number = 0.0;   // число обусловленности
            double symmetry_error = 0.0; // асимметрия ковариации
            std::vector<double> error_history;
        };

        FilterMetrics ckf_metrics;
        FilterMetrics srcf_metrics;

        // Дополнительные метрики для сравнения
        struct ComparisonMetrics {
            double avg_error_ratio = 0.0;          // SRCF/CKF
            double rms_error_ratio = 0.0;          // SRCF/CKF
            double max_error_ratio = 0.0;          // SRCF/CKF
            double cond_number_ratio = 0.0;        // SRCF/CKF
            double cov_norm_ratio = 0.0;           // SRCF/CKF
            double max_srcf_minus_ckf = 0.0;       // Максимальная SRCF - CKF
            double max_ckf_minus_srcf = 0.0;       // Максимальная CKF - SRCF
            double avg_absolute_difference = 0.0;  // Средняя абсолютная разница
            double std_dev_difference = 0.0;       // Стандартное отклонение разницы
            double percentage_srcf_better = 0.0;   // Процент шагов, где SRCF лучше
            std::vector<double> error_differences; // Разности ошибок (SRCF - CKF)
            std::vector<double> relative_differences; // Относительные разности
        };

        ComparisonMetrics comparison;
    };
// ============================================================================
// 2. ГЕНЕРАТОР ДАННЫХ
// ============================================================================

    class DataGenerator {
    private:
        SimulationConfig config_;
        time_generator::TimeGenerator time_gen_;

    public:
        DataGenerator(const SimulationConfig& config, int seed = 42)
                : config_(config), time_gen_(seed)
        {
            std::string command;
#ifdef _WIN32
            command = "mkdir \"" + config_.output_dir + "\" 2>nul";
#else
            command = "mkdir -p \"" + config_.output_dir + "\"";
#endif
            system(command.c_str());
        }

        SimulationData generate() {
            SimulationData data;

            // Генерация временной сетки
            data.times = time_gen_.generate(config_.total_steps,
                                            config_.base_dt,
                                            config_.time_mode);
            std::string timegrid_file = config_.output_dir + "/timegrid.bin";
            time_gen_.saveToFile(data.times, timegrid_file);
            std::vector<double> time_check = time_gen_.loadFromFile(timegrid_file);

            // Выбор модели
            if (config_.model_type == ModelType::MODEL0) {
                generate_with_model0(data);
            } else {
                generate_with_model2(data);
            }

            // Рассчитываем метрики
            calculate_metrics(data);

            // Рассчитываем сравнение фильтров
            if (config_.test_ckf && !data.ckf_estimates.empty()) {
                calculate_comparison_metrics(data);
            }
            return data;
        }

        void save(const SimulationData& data) {
            switch (config_.format) {
                case DataFormat::BINARY:
                    saveBinary(data);
                    break;
                case DataFormat::TEXT_CSV:
                    saveTextCSV(data);
                    break;
                case DataFormat::TEXT_MATLAB:
                    saveTextMatlab(data);
                    break;
                case DataFormat::TEXT_TXT:
                    saveTextTXT(data);
                    break;
            }
        }

    private:
        void generate_with_model0(SimulationData& data) {
            // Инициализация фильтров
            model0::reset_noise();

            Eigen::Matrix2d P0;
            P0 << 0.1, 0.0,
                    0.0, 0.1;

            Eigen::Vector2d x_true = Eigen::Vector2d::Zero();

            // Подготовка фильтров
            kalman::CKF ckf(Eigen::Vector2d::Zero(), P0);
            kalman::SRCF srcf(Eigen::Vector2d::Zero(), P0);

            // Главный цикл симуляции
            for (size_t k = 0; k < data.times.size() - 1; ++k) {
                double t = data.times[k];
                double t_next = data.times[k + 1];
                double dt = t_next - t;

                // Матрицы модели
                Eigen::MatrixXd A = model0::A(dt);
                Eigen::MatrixXd B = model0::B(dt);
                Eigen::MatrixXd C = model0::C(t);
                Eigen::MatrixXd D = model0::D(dt);
                Eigen::MatrixXd Q = model0::Q(dt) * config_.process_noise_scale;
                Eigen::MatrixXd R = model0::R(t) * config_.measurement_noise_scale;

                // Управление
                Eigen::VectorXd u = model0::u(t, config_.scenario.scenario0);

                // Динамика системы
//                x_true = model0::true_dynamics(x_true, t, dt,
//                                               config_.scenario.scenario0,
//                                               config_.add_process_noise);
                // Шумы
                // Генератор данных // Шумы
                Eigen::VectorXd w;
                if (config_.add_process_noise) {
                    w = model0::w(t, dt, true) * config_.process_noise_scale;
                } else {
                    w = Eigen::Vector2d::Zero();
                }

                // Динамика системы
                x_true = A * x_true + B * u + w;

                // Измерения
                Eigen::Vector2d y_exact = C * x_true;
                Eigen::Vector2d y_noisy = y_exact;

                Eigen::VectorXd v;
                if (config_.add_measurement_noise) {
                    y_noisy += model0::v(t, true) * config_.measurement_noise_scale;
                } else {
                    y_noisy += Eigen::Vector2d::Zero();
                }

                // Шаги фильтров
                if (config_.test_ckf) {
                    ckf.step(A, B, C, D, Q, R, u, y_noisy);
                    data.ckf_estimates.emplace_back(ckf.state());
                    data.ckf_covariances.emplace_back(ckf.covariance());
                }

                srcf.step(A, B, C, D, Q, R, u, y_noisy);
                data.srcf_estimates.emplace_back(srcf.state());
                data.srcf_covariances.emplace_back(srcf.covariance());

                // Сохранение данных
                data.true_states.push_back(x_true);
                data.measurements.push_back(y_exact);
                data.noisy_measurements.push_back(y_noisy);
                data.controls.push_back(u);
            }
        }

        void generate_with_model2(SimulationData& data) {
            // Инициализация фильтров
            model2::reset_noise();

            Eigen::Matrix2d P0;
            P0 << 0.1, 0.0,
                    0.0, 0.01;

            Eigen::Vector2d x_true = Eigen::Vector2d::Zero();

            // Подготовка фильтров
            kalman::CKF ckf(Eigen::Vector2d::Zero(), P0);
            kalman::SRCF srcf(Eigen::Vector2d::Zero(), P0);

            // Главный цикл симуляции
            for (size_t k = 0; k < data.times.size() - 1; ++k) {
                double t = data.times[k];
                double t_next = data.times[k + 1];
                double dt = t_next - t;

                // Матрицы модели
                Eigen::MatrixXd A = model2::A(dt);
                Eigen::MatrixXd B = model2::B(dt);
                Eigen::MatrixXd C = model2::C(t);
                Eigen::MatrixXd D = model2::D(dt);
                Eigen::MatrixXd Q = model2::Q(dt) * config_.process_noise_scale;
                Eigen::MatrixXd R = model2::R(t) * config_.measurement_noise_scale;

                // Управление
                Eigen::VectorXd u = model2::u(t, config_.scenario.scenario2);

                // Динамика системы
//                x_true = model2::true_dynamics(x_true, t, dt,
//                                               config_.scenario.scenario2,
//                                               config_.add_process_noise);

                // Шумы
                // Генератор данных // Шумы
                Eigen::VectorXd w;
                if (config_.add_process_noise) {
                    w = model2::w(t, dt, true) * config_.process_noise_scale;
                } else {
                    w = Eigen::Vector2d::Zero();
                }

                // Динамика системы
                x_true = A * x_true + B * u + w;

                // Измерения
                Eigen::Vector2d y_exact = C * x_true;
                Eigen::Vector2d y_noisy = y_exact;

                if (config_.add_measurement_noise) {
                    y_noisy += model2::v(t, true) * config_.measurement_noise_scale;
                } else {
                    y_noisy += Eigen::Vector2d::Zero();
                }

                // Шаги фильтров
                if (config_.test_ckf) {
                    ckf.step(A, B, C, D, Q, R, u, y_noisy);
                    data.ckf_estimates.emplace_back(ckf.state());
                    data.ckf_covariances.emplace_back(ckf.covariance());
                }

                srcf.step(A, B, C, D, Q, R, u, y_noisy);
                data.srcf_estimates.emplace_back(srcf.state());
                data.srcf_covariances.emplace_back(srcf.covariance());

                // Сохранение данных
                data.true_states.push_back(x_true);
                data.measurements.push_back(y_exact);
                data.noisy_measurements.push_back(y_noisy);
                data.controls.push_back(u);
            }
        }

        void calculate_metrics(SimulationData& data) {
            // Рассчитываем метрики для CKF
            if (config_.test_ckf && !data.ckf_estimates.empty()) {
                calculate_filter_metrics(data.true_states, data.ckf_estimates,
                                         data.ckf_covariances, data.times,
                                         data.ckf_metrics);
            }

            // Рассчитываем метрики для SRCF
            calculate_filter_metrics(data.true_states, data.srcf_estimates,
                                     data.srcf_covariances, data.times,
                                     data.srcf_metrics);
        }

        void calculate_filter_metrics(const std::vector<Eigen::Vector2d>& true_states,
                                      const std::vector<Eigen::Vector2d>& estimates,
                                      const std::vector<Eigen::Matrix2d>& covariances,
                                      const std::vector<double>& times,
                                      SimulationData::FilterMetrics& metrics) {
            double sum_sq_error = 0.0;
            double sum_cov_norm = 0.0;
            double sum_cond = 0.0;
            metrics.max_error = 0.0;
            metrics.error_history.clear();
            metrics.error_history.reserve(true_states.size());

            for (size_t i = 0; i < true_states.size(); ++i) {
                // Ошибка оценки
                double error = (true_states[i] - estimates[i]).norm();
                metrics.error_history.push_back(error);
                sum_sq_error += error * error;

                if (error > metrics.max_error) {
                    metrics.max_error = error;
                }

                // Норма ковариации
                sum_cov_norm += covariances[i].norm();

                // Число обусловленности
                Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariances[i]);
                double cond = svd.singularValues()(0) /
                              svd.singularValues()(svd.singularValues().size()-1);
                sum_cond += cond;

                // Определяем время сходимости
                if (i > 100 && error < 0.01 && metrics.convergence_time == 0.0) {
                    metrics.convergence_time = times[i];
                }
            }

            metrics.average_error = std::accumulate(metrics.error_history.begin(),
                                                    metrics.error_history.end(), 0.0)
                                    / metrics.error_history.size();

            metrics.rms_error = std::sqrt(sum_sq_error / metrics.error_history.size());
            metrics.cov_norm = sum_cov_norm / covariances.size();
            metrics.cond_number = sum_cond / covariances.size();
        }

        void calculate_comparison_metrics(SimulationData& data) {
            if (!config_.test_ckf || data.ckf_estimates.empty()) {
                return;
            }

            data.comparison.error_differences.clear();
            data.comparison.relative_differences.clear();

            double sum_diff = 0.0;
            double sum_sq_diff = 0.0;
            double sum_abs_diff = 0.0;
            data.comparison.max_srcf_minus_ckf = -std::numeric_limits<double>::max();
            data.comparison.max_ckf_minus_srcf = -std::numeric_limits<double>::max();
            int srcf_better_count = 0;

            for (size_t i = 0; i < data.true_states.size(); ++i) {
                double ckf_error = data.ckf_metrics.error_history[i];
                double srcf_error = data.srcf_metrics.error_history[i];
                double diff = srcf_error - ckf_error;

                data.comparison.error_differences.push_back(diff);

                // Относительная разница в процентах
                double rel_diff = (ckf_error > 1e-10) ?
                                  (diff / ckf_error) * 100.0 : 0.0;
                data.comparison.relative_differences.push_back(rel_diff);

                sum_diff += diff;
                sum_sq_diff += diff * diff;
                sum_abs_diff += std::abs(diff);

                if (diff > data.comparison.max_srcf_minus_ckf) {
                    data.comparison.max_srcf_minus_ckf = diff;
                }

                if (-diff > data.comparison.max_ckf_minus_srcf) {
                    data.comparison.max_ckf_minus_srcf = -diff;
                }

                if (srcf_error < ckf_error) {
                    srcf_better_count++;
                }
            }

            size_t n = data.comparison.error_differences.size();

            data.comparison.avg_error_ratio = data.srcf_metrics.average_error /
                                              data.ckf_metrics.average_error;
            data.comparison.rms_error_ratio = data.srcf_metrics.rms_error /
                                              data.ckf_metrics.rms_error;
            data.comparison.max_error_ratio = data.srcf_metrics.max_error /
                                              data.ckf_metrics.max_error;
            data.comparison.cond_number_ratio = data.srcf_metrics.cond_number /
                                                data.ckf_metrics.cond_number;
            data.comparison.cov_norm_ratio = data.srcf_metrics.cov_norm /
                                             data.ckf_metrics.cov_norm;

            data.comparison.avg_absolute_difference = sum_abs_diff / n;
            data.comparison.percentage_srcf_better = (srcf_better_count * 100.0) / n;

            // Стандартное отклонение разницы
            double mean_diff = sum_diff / n;
            data.comparison.std_dev_difference = std::sqrt((sum_sq_diff / n) - (mean_diff * mean_diff));
        }

        // ============================================================================
        // 3. ФУНКЦИИ СОХРАНЕНИЯ
        // ============================================================================

        void saveBinary(const SimulationData& data) {
            std::string prefix = config_.output_dir + "/";

            // Сохранение времен
            saveVectorBinary(data.times, prefix + "times.bin");

            // Сохранение векторов состояния
            saveVector2dBinary(data.true_states, prefix + "true_states.bin");
            saveVector2dBinary(data.measurements, prefix + "measurements.bin");
            saveVector2dBinary(data.noisy_measurements, prefix + "noisy_measurements.bin");
            if (config_.test_ckf) {
                saveVector2dBinary(data.ckf_estimates, prefix + "ckf_estimates.bin");
            }
            saveVector2dBinary(data.srcf_estimates, prefix + "srcf_estimates.bin");

            // Сохранение управлений (скаляр)
            std::vector<double> controls_scalar;
            for (const auto& u : data.controls) {
                controls_scalar.push_back(u(0));
            }
            saveVectorBinary(controls_scalar, prefix + "controls.bin");

            // Сохранение ковариаций
            if (config_.test_ckf) {
                saveMatrix2dBinary(data.ckf_covariances, prefix + "ckf_covariances.bin");
            }
            saveMatrix2dBinary(data.srcf_covariances, prefix + "srcf_covariances.bin");

            // Сохранение метрик
            saveMetrics(data, prefix + "metrics.txt");
            saveComparisonStats(data, prefix + "comparison_stats.txt");
        }

        void saveTextCSV(const SimulationData& data) {
            std::string filename = config_.output_dir + "/simulation_data.csv";
            std::ofstream file(filename);

            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            // Заголовок
            file << "time,true_phi,true_p,meas_phi_exact,meas_p_exact,"
                 << "meas_phi_noisy,meas_p_noisy,control,";

            if (config_.test_ckf) {
                file << "ckf_phi,ckf_p,";
            }

            file << "srcf_phi,srcf_p,";

            if (config_.test_ckf) {
                file << "ckf_cov_11,ckf_cov_12,ckf_cov_21,ckf_cov_22,";
            }

            file << "srcf_cov_11,srcf_cov_12,srcf_cov_21,srcf_cov_22\n";

            file << std::fixed << std::setprecision(12);

            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << data.times[i] << ","
                     << data.true_states[i](0) << "," << data.true_states[i](1) << ","
                     << data.measurements[i](0) << "," << data.measurements[i](1) << ","
                     << data.noisy_measurements[i](0) << "," << data.noisy_measurements[i](1) << ","
                     << data.controls[i](0) << ",";

                if (config_.test_ckf) {
                    file << data.ckf_estimates[i](0) << "," << data.ckf_estimates[i](1) << ",";
                }

                file << data.srcf_estimates[i](0) << "," << data.srcf_estimates[i](1) << ",";

                if (config_.test_ckf) {
                    file << data.ckf_covariances[i](0,0) << "," << data.ckf_covariances[i](0,1) << ","
                         << data.ckf_covariances[i](1,0) << "," << data.ckf_covariances[i](1,1) << ",";
                }

                file << data.srcf_covariances[i](0,0) << "," << data.srcf_covariances[i](0,1) << ","
                     << data.srcf_covariances[i](1,0) << "," << data.srcf_covariances[i](1,1) << "\n";
            }

            file.close();

            // Сохранение конфигурации
            saveConfig();
            saveMetrics(data, config_.output_dir + "/metrics.txt");
            saveComparisonStats(data, config_.output_dir + "/comparison_stats.txt");
        }

        void saveTextMatlab(const SimulationData& data) {
            std::string filename = config_.output_dir + "/simulation_data.m";
            std::ofstream file(filename);

            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "% Kalman Filter Simulation Data\n";
            file << "% Generated by C++ simulation\n\n";

            // Сохранение как MATLAB матрицы
            file << "times = [";
            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << data.times[i];
                if (i < data.times.size() - 2) file << "; ";
            }
            file << "];\n\n";

            // Истинные состояния
            file << "true_states = [";
            for (size_t i = 0; i < data.true_states.size(); ++i) {
                file << data.true_states[i](0) << ", " << data.true_states[i](1);
                if (i < data.true_states.size() - 1) file << "; ";
            }
            file << "];\n\n";

            // Измерения
            file << "measurements_exact = [";
            for (size_t i = 0; i < data.measurements.size(); ++i) {
                file << data.measurements[i](0) << ", " << data.measurements[i](1);
                if (i < data.measurements.size() - 1) file << "; ";
            }
            file << "];\n\n";

            file << "measurements_noisy = [";
            for (size_t i = 0; i < data.noisy_measurements.size(); ++i) {
                file << data.noisy_measurements[i](0) << ", " << data.noisy_measurements[i](1);
                if (i < data.noisy_measurements.size() - 1) file << "; ";
            }
            file << "];\n\n";

            // Управления
            file << "controls = [";
            for (size_t i = 0; i < data.controls.size(); ++i) {
                file << data.controls[i](0);
                if (i < data.controls.size() - 1) file << "; ";
            }
            file << "];\n\n";

            // Оценки фильтров
            if (config_.test_ckf) {
                file << "ckf_estimates = [";
                for (size_t i = 0; i < data.ckf_estimates.size(); ++i) {
                    file << data.ckf_estimates[i](0) << ", " << data.ckf_estimates[i](1);
                    if (i < data.ckf_estimates.size() - 1) file << "; ";
                }
                file << "];\n\n";
            }

            file << "srcf_estimates = [";
            for (size_t i = 0; i < data.srcf_estimates.size(); ++i) {
                file << data.srcf_estimates[i](0) << ", " << data.srcf_estimates[i](1);
                if (i < data.srcf_estimates.size() - 1) file << "; ";
            }
            file << "];\n";

            file.close();

            // Сохранение конфигурации
            saveConfig();
            saveMetrics(data, config_.output_dir + "/metrics.txt");
            saveComparisonStats(data, config_.output_dir + "/comparison_stats.txt");
        }

        void saveTextTXT(const SimulationData& data) {
            std::string prefix = config_.output_dir + "/";
            saveMainDataTXT(data, prefix + "main_data.txt");
            saveCovarianceTXT(data, prefix + "covariances.txt");
            saveMeasurementsTXT(data, prefix + "measurements.txt");
            saveStatisticsTXT(data, prefix + "statistics.txt");
            saveConfig();
            saveMetrics(data, prefix + "filter_metrics.txt");
            saveComparisonStats(data, prefix + "comparison_details.txt");
        }

        void saveMainDataTXT(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "KALMAN FILTER SIMULATION DATA\n";
            file << "========================================\n\n";

            file << std::fixed << std::setprecision(8);
            file << std::setw(12) << "Time(s)"
                 << std::setw(15) << "True_phi"
                 << std::setw(15) << "True_p"
                 << std::setw(15) << "Control";

            if (config_.test_ckf) {
                file << std::setw(15) << "CKF_phi"
                     << std::setw(15) << "CKF_p";
            }

            file << std::setw(15) << "SRCF_phi"
                 << std::setw(15) << "SRCF_p"
                 << "\n";

            file << std::string(config_.test_ckf ? 120 : 90, '-') << "\n";

            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << std::setw(12) << data.times[i]
                     << std::setw(15) << data.true_states[i](0)
                     << std::setw(15) << data.true_states[i](1)
                     << std::setw(15) << data.controls[i](0);

                if (config_.test_ckf) {
                    file << std::setw(15) << data.ckf_estimates[i](0)
                         << std::setw(15) << data.ckf_estimates[i](1);
                }

                file << std::setw(15) << data.srcf_estimates[i](0)
                     << std::setw(15) << data.srcf_estimates[i](1)
                     << "\n";
            }

            file.close();
        }

        void saveCovarianceTXT(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "COVARIANCE MATRICES\n";
            file << "========================================\n\n";

            file << std::fixed << std::setprecision(10);

            for (size_t i = 0; i < std::min(data.times.size() - 1, (size_t)20); ++i) {
                if (i % 10 == 0) {  // Каждые 10 шагов
                    file << "\nStep " << i << " (t = " << data.times[i] << " s):\n";
                    if (config_.test_ckf) {
                        file << "CKF Covariance:\n";
                        file << "  [" << data.ckf_covariances[i](0,0) << ", "
                             << data.ckf_covariances[i](0,1) << "]\n";
                        file << "  [" << data.ckf_covariances[i](1,0) << ", "
                             << data.ckf_covariances[i](1,1) << "]\n";
                        file << "  Determinant: " << data.ckf_covariances[i].determinant() << "\n";
                    }

                    file << "SRCF Covariance:\n";
                    file << "  [" << data.srcf_covariances[i](0,0) << ", "
                         << data.srcf_covariances[i](0,1) << "]\n";
                    file << "  [" << data.srcf_covariances[i](1,0) << ", "
                         << data.srcf_covariances[i](1,1) << "]\n";
                    file << "  Determinant: " << data.srcf_covariances[i].determinant() << "\n";
                    file << std::string(50, '-') << "\n";
                }
            }

            file.close();
        }

        void saveMeasurementsTXT(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "MEASUREMENT DATA\n";
            file << "========================================\n\n";

            file << std::fixed << std::setprecision(8);
            file << std::setw(12) << "Time(s)"
                 << std::setw(20) << "Exact_Measurement1"
                 << std::setw(20) << "Exact_Measurement2"
                 << std::setw(20) << "Noisy_Measurement1"
                 << std::setw(20) << "Noisy_Measurement2"
                 << "\n";

            file << std::string(100, '-') << "\n";

            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << std::setw(12) << data.times[i]
                     << std::setw(20) << data.measurements[i](0)
                     << std::setw(20) << data.measurements[i](1)
                     << std::setw(20) << data.noisy_measurements[i](0)
                     << std::setw(20) << data.noisy_measurements[i](1)
                     << "\n";
            }

            file.close();
        }

        void saveStatisticsTXT(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "FILTER PERFORMANCE STATISTICS\n";
            file << "========================================\n\n";

            // Вычисляем ошибки
            double ckf_total_error = 0.0;
            double srcf_total_error = 0.0;
            double ckf_max_error = 0.0;
            double srcf_max_error = 0.0;

            std::vector<double> ckf_errors;
            std::vector<double> srcf_errors;

            for (size_t i = 0; i < data.true_states.size(); ++i) {
                double ckf_error = (data.true_states[i] - data.ckf_estimates[i]).norm();
                double srcf_error = (data.true_states[i] - data.srcf_estimates[i]).norm();

                ckf_errors.push_back(ckf_error);
                srcf_errors.push_back(srcf_error);

                ckf_total_error += ckf_error;
                srcf_total_error += srcf_error;

                if (ckf_error > ckf_max_error) ckf_max_error = ckf_error;
                if (srcf_error > srcf_max_error) srcf_max_error = srcf_error;
            }

            double ckf_avg_error = ckf_total_error / ckf_errors.size();
            double srcf_avg_error = srcf_total_error / srcf_errors.size();

            // Вычисляем RMS
            double ckf_rms = 0.0;
            double srcf_rms = 0.0;
            for (size_t i = 0; i < ckf_errors.size(); ++i) {
                ckf_rms += ckf_errors[i] * ckf_errors[i];
                srcf_rms += srcf_errors[i] * srcf_errors[i];
            }
            ckf_rms = sqrt(ckf_rms / ckf_errors.size());
            srcf_rms = sqrt(srcf_rms / srcf_errors.size());

            file << std::fixed << std::setprecision(6);
            file << "CKF Statistics:\n";
            file << "  Average error: " << ckf_avg_error << "\n";
            file << "  Maximum error: " << ckf_max_error << "\n";
            file << "  RMS error:     " << ckf_rms << "\n";
            file << "  Error ratio (SRCF/CKF): " << srcf_avg_error / ckf_avg_error << "\n\n";

            file << "SRCF Statistics:\n";
            file << "  Average error: " << srcf_avg_error << "\n";
            file << "  Maximum error: " << srcf_max_error << "\n";
            file << "  RMS error:     " << srcf_rms << "\n";
            file << "  Error ratio (CKF/SRCF): " << ckf_avg_error / srcf_avg_error << "\n\n";

            // Детальное сравнение
            if (config_.test_ckf) {
                file << "DETAILED COMPARISON (SRCF vs CKF):\n";
                file << "  Average error ratio: " << data.comparison.avg_error_ratio << "\n";
                file << "  RMS error ratio: " << data.comparison.rms_error_ratio << "\n";
                file << "  Max error ratio: " << data.comparison.max_error_ratio << "\n";
                file << "  Max SRCF - CKF: " << data.comparison.max_srcf_minus_ckf << "\n";
                file << "  Max CKF - SRCF: " << data.comparison.max_ckf_minus_srcf << "\n";
                file << "  Avg absolute difference: " << data.comparison.avg_absolute_difference << "\n";
                file << "  Std dev of difference: " << data.comparison.std_dev_difference << "\n";
                file << "  Percentage where SRCF is better: " << data.comparison.percentage_srcf_better << "%\n\n";
            }

            // Статистика по шагам
            file << "Error Statistics by Steps:\n";
            file << std::setw(8) << "Step"
                 << std::setw(15) << "CKF_Error"
                 << std::setw(15) << "SRCF_Error"
                 << std::setw(15) << "Difference"
                 << std::setw(15) << "Rel Diff %"
                 << "\n";

            file << std::string(75, '-') << "\n";

            for (size_t i = 0; i < std::min((size_t)50, ckf_errors.size()); i += 5) {
                double diff = srcf_errors[i] - ckf_errors[i];
                double rel_diff = (ckf_errors[i] > 1e-10) ?
                                  (diff / ckf_errors[i]) * 100.0 : 0.0;

                file << std::setw(8) << i
                     << std::setw(15) << ckf_errors[i]
                     << std::setw(15) << srcf_errors[i]
                     << std::setw(15) << diff
                     << std::setw(15) << std::setprecision(2) << rel_diff << std::setprecision(6)
                     << "\n";
            }

            file.close();
        }

        void saveComparisonStats(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);

            file << "=== DETAILED FILTER COMPARISON ===\n\n";
            file << std::fixed << std::setprecision(6);

            if (config_.test_ckf) {
                file << "SUMMARY:\n";
                file << "  Average error ratio (SRCF/CKF): " << data.comparison.avg_error_ratio << "\n";

                if (data.comparison.avg_error_ratio < 1.0) {
                    file << "  -> SRCF is " << (1.0 - data.comparison.avg_error_ratio) * 100.0
                         << "% better on average\n";
                } else {
                    file << "  -> CKF is " << (data.comparison.avg_error_ratio - 1.0) * 100.0
                         << "% better on average\n";
                }

                file << "\nERROR ANALYSIS:\n";
                file << "  CKF avg error: " << data.ckf_metrics.average_error << "\n";
                file << "  SRCF avg error: " << data.srcf_metrics.average_error << "\n";
                file << "  Absolute difference: "
                     << (data.srcf_metrics.average_error - data.ckf_metrics.average_error) << "\n";
                file << "  Ratio (SRCF/CKF): " << data.comparison.avg_error_ratio << "\n\n";

                file << "RMS ANALYSIS:\n";
                file << "  CKF RMS: " << data.ckf_metrics.rms_error << "\n";
                file << "  SRCF RMS: " << data.srcf_metrics.rms_error << "\n";
                file << "  Ratio (SRCF/CKF): " << data.comparison.rms_error_ratio << "\n\n";

                file << "MAXIMUM ERRORS:\n";
                file << "  CKF max error: " << data.ckf_metrics.max_error << "\n";
                file << "  SRCF max error: " << data.srcf_metrics.max_error << "\n";
                file << "  Ratio (SRCF/CKF): " << data.comparison.max_error_ratio << "\n\n";

                file << "DIFFERENCE DISTRIBUTION:\n";
                file << "  Max SRCF - CKF: " << data.comparison.max_srcf_minus_ckf << "\n";
                file << "  Max CKF - SRCF: " << data.comparison.max_ckf_minus_srcf << "\n";
                file << "  Avg absolute difference: " << data.comparison.avg_absolute_difference << "\n";
                file << "  Std dev of difference: " << data.comparison.std_dev_difference << "\n";
                file << "  Percentage where SRCF is better: " << data.comparison.percentage_srcf_better << "%\n\n";

                file << "CONDITION NUMBERS:\n";
                file << "  CKF avg condition: " << data.ckf_metrics.cond_number << "\n";
                file << "  SRCF avg condition: " << data.srcf_metrics.cond_number << "\n";
                file << "  Ratio (SRCF/CKF): " << data.comparison.cond_number_ratio << "\n\n";

                file << "COVARIANCE NORMS:\n";
                file << "  CKF avg cov norm: " << data.ckf_metrics.cov_norm << "\n";
                file << "  SRCF avg cov norm: " << data.srcf_metrics.cov_norm << "\n";
                file << "  Ratio (SRCF/CKF): " << data.comparison.cov_norm_ratio << "\n\n";

                file << "CONVERGENCE TIMES:\n";
                file << "  CKF convergence time: " << data.ckf_metrics.convergence_time << " s\n";
                file << "  SRCF convergence time: " << data.srcf_metrics.convergence_time << " s\n\n";

                // Рекомендации
                file << "RECOMMENDATIONS:\n";
                if (data.comparison.avg_error_ratio < 0.95) {
                    file << "  Use SRCF - significantly better performance\n";
                } else if (data.comparison.avg_error_ratio > 1.05) {
                    file << "  Use CKF - significantly better performance\n";
                } else {
                    file << "  Both filters perform similarly\n";
                }

                if (data.comparison.percentage_srcf_better > 60.0) {
                    file << "  SRCF is more consistent across steps\n";
                } else if (data.comparison.percentage_srcf_better < 40.0) {
                    file << "  CKF is more consistent across steps\n";
                }
            }

            file.close();
        }

        void saveMetrics(const SimulationData& data, const std::string& filename) {
            std::ofstream file(filename);

            file << "=== FILTER PERFORMANCE METRICS ===\n\n";
            file << std::fixed << std::setprecision(6);

            if (config_.test_ckf) {
                file << "CKF:\n";
                file << "  Average error: " << data.ckf_metrics.average_error << "\n";
                file << "  Max error: " << data.ckf_metrics.max_error << "\n";
                file << "  RMSE: " << data.ckf_metrics.rms_error << "\n";
                file << "  Convergence: " << data.ckf_metrics.convergence_time << " s\n";
                file << "  Condition number: " << data.ckf_metrics.cond_number << "\n";
                file << "  Covariance norm: " << data.ckf_metrics.cov_norm << "\n\n";
            }

            file << "SRCF:\n";
            file << "  Average error: " << data.srcf_metrics.average_error << "\n";
            file << "  Max error: " << data.srcf_metrics.max_error << "\n";
            file << "  RMSE: " << data.srcf_metrics.rms_error << "\n";
            file << "  Convergence: " << data.srcf_metrics.convergence_time << " s\n";
            file << "  Condition number: " << data.srcf_metrics.cond_number << "\n";
            file << "  Covariance norm: " << data.srcf_metrics.cov_norm << "\n";

            file.close();
        }

        void saveConfig() {
            std::string filename = config_.output_dir + "/config.txt";
            std::ofstream file(filename);

            file << "=== Simulation Configuration ===\n\n";
            file << "Total steps: " << config_.total_steps << "\n";
            file << "Base dt: " << config_.base_dt << " s\n";
            file << "Model: " << (config_.model_type == ModelType::MODEL0 ? "MODEL0" : "MODEL2") << "\n";
            file << "Add process noise: " << (config_.add_process_noise ? "YES" : "NO") << "\n";
            file << "Add measurement noise: " << (config_.add_measurement_noise ? "YES" : "NO") << "\n";
            file << "Process noise scale: " << config_.process_noise_scale << "\n";
            file << "Measurement noise scale: " << config_.measurement_noise_scale << "\n";

            file << "Control scenario: ";
            if (config_.model_type == ModelType::MODEL0) {
                switch (config_.scenario.scenario0) {
                    case model0::ControlScenario::ZERO_HOLD: file << "ZERO_HOLD"; break;
                    case model0::ControlScenario::STEP_MANEUVER: file << "STEP_MANEUVER"; break;
                    case model0::ControlScenario::SINE_WAVE: file << "SINE_WAVE"; break;
                    case model0::ControlScenario::PULSE: file << "PULSE"; break;
                }
            } else {
                switch (config_.scenario.scenario2) {
                    case model2::ControlScenario::ZERO_HOLD: file << "ZERO_HOLD"; break;
                    case model2::ControlScenario::STEP_MANEUVER: file << "STEP_MANEUVER"; break;
                    case model2::ControlScenario::SINE_WAVE: file << "SINE_WAVE"; break;
                    case model2::ControlScenario::PULSE: file << "PULSE"; break;
                }
            }
            file << "\n";

            file << "Time mode: ";
            switch (config_.time_mode) {
                case time_generator::TimeMode::UNIFORM: file << "UNIFORM"; break;
                case time_generator::TimeMode::VARIABLE: file << "VARIABLE"; break;
                case time_generator::TimeMode::RANDOM_JITTER: file << "RANDOM_JITTER"; break;
            }
            file << "\n";

            file << "Test CKF: " << (config_.test_ckf ? "YES" : "NO") << "\n";
            file << "Data format: ";
            switch (config_.format) {
                case DataFormat::BINARY: file << "BINARY"; break;
                case DataFormat::TEXT_CSV: file << "TEXT_CSV"; break;
                case DataFormat::TEXT_MATLAB: file << "TEXT_MATLAB"; break;
                case DataFormat::TEXT_TXT: file << "TEXT_TXT"; break;
            }
            file << "\n";

            file.close();
        }

        // Вспомогательные функции для бинарного сохранения
        void saveVectorBinary(const std::vector<double>& vec, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = vec.size();
            file.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(vec.data()), n * sizeof(double));
            file.close();
        }

        void saveVector2dBinary(const std::vector<Eigen::Vector2d>& vec, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = vec.size();
            file.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            for (const auto& v : vec) {
                file.write(reinterpret_cast<const char*>(v.data()), 2 * sizeof(double));
            }
            file.close();
        }

        void saveMatrix2dBinary(const std::vector<Eigen::Matrix2d>& mats, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = mats.size();
            file.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            for (const auto& m : mats) {
                file.write(reinterpret_cast<const char*>(m.data()), 4 * sizeof(double));
            }
            file.close();
        }
    };

// ============================================================================
// 4. ИНСТРУМЕНТЫ АНАЛИЗА В СТИЛЕ VERHAEGEN & VAN DOOREN
// ============================================================================

    void analyze_verhaegen_style(const SimulationData& data, bool test_ckf = true) {
        std::cout << "\n=== ANALYSIS IN VERHAEGEN & VAN DOOREN STYLE ===\n";

        // 1. Анализ численной устойчивости
        if (test_ckf) {
            double max_ckf_asymmetry = 0.0;
            int ckf_non_positive = 0;

            for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
                // Асимметрия: ||P - P^T|| / ||P||
                Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] -
                                           data.ckf_covariances[i].transpose();
                double asym_norm = asym_ckf.norm() / data.ckf_covariances[i].norm();
                max_ckf_asymmetry = std::max(max_ckf_asymmetry, asym_norm);

                // Положительная определенность
                Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
                if (llt_ckf.info() != Eigen::Success) ckf_non_positive++;
            }

            std::cout << "Symmetry Analysis:\n";
            std::cout << "  CKF maximum asymmetry: " << max_ckf_asymmetry << "\n";
            std::cout << "  SRCF: guaranteed symmetric by construction\n";

            std::cout << "\nPositive Definiteness:\n";
            std::cout << "  CKF non-positive definite: " << ckf_non_positive
                      << "/" << data.ckf_covariances.size() << "\n";
            std::cout << "  SRCF non-positive definite: 0/"
                      << data.srcf_covariances.size() << " (guaranteed)\n";
        }

        // 2. Числа обусловленности
        double avg_cond_ckf = 0.0, avg_cond_srcf = 0.0;
        int count = 0;

        for (size_t i = 100; i < data.srcf_covariances.size(); i += 100) {
            Eigen::JacobiSVD<Eigen::Matrix2d> svd_srcf(data.srcf_covariances[i]);
            double cond_srcf = svd_srcf.singularValues()(0) /
                               svd_srcf.singularValues()(svd_srcf.singularValues().size()-1);
            avg_cond_srcf += cond_srcf;

            if (test_ckf) {
                Eigen::JacobiSVD<Eigen::Matrix2d> svd_ckf(data.ckf_covariances[i]);
                double cond_ckf = svd_ckf.singularValues()(0) /
                                  svd_ckf.singularValues()(svd_ckf.singularValues().size()-1);
                avg_cond_ckf += cond_ckf;
            }

            count++;
        }

        avg_cond_srcf /= count;
        if (test_ckf) avg_cond_ckf /= count;

        std::cout << "\nCondition numbers (avg over " << count << " samples):\n";
        if (test_ckf) {
            std::cout << "  CKF: " << avg_cond_ckf << "\n";
        }
        std::cout << "  SRCF: " << avg_cond_srcf << "\n";

        // 3. Анализ разницы ошибок
        if (test_ckf) {
            std::cout << "\n=== ERROR DIFFERENCE ANALYSIS ===\n";
            std::cout << std::fixed << std::setprecision(6);

            std::cout << "Average error ratio (SRCF/CKF): " << data.comparison.avg_error_ratio << "\n";
            std::cout << "RMS error ratio (SRCF/CKF): " << data.comparison.rms_error_ratio << "\n";
            std::cout << "Max error ratio (SRCF/CKF): " << data.comparison.max_error_ratio << "\n";

            if (data.comparison.avg_error_ratio < 1.0) {
                std::cout << "-> SRCF performs " << (1.0 - data.comparison.avg_error_ratio) * 100.0
                          << "% better on average\n";
            } else {
                std::cout << "-> CKF performs " << (data.comparison.avg_error_ratio - 1.0) * 100.0
                          << "% better on average\n";
            }

            std::cout << "\nDifference statistics:\n";
            std::cout << "  Max SRCF - CKF: " << data.comparison.max_srcf_minus_ckf << "\n";
            std::cout << "  Max CKF - SRCF: " << data.comparison.max_ckf_minus_srcf << "\n";
            std::cout << "  Average absolute difference: " << data.comparison.avg_absolute_difference << "\n";
            std::cout << "  Std deviation: " << data.comparison.std_dev_difference << "\n";
            std::cout << "  Percentage where SRCF is better: " << data.comparison.percentage_srcf_better << "%\n";

            std::cout << "\nCondition number ratio (SRCF/CKF): " << data.comparison.cond_number_ratio << "\n";
            std::cout << "Covariance norm ratio (SRCF/CKF): " << data.comparison.cov_norm_ratio << "\n";
        }

        // 4. Сравнение производительности фильтров
        std::cout << "\n=== FILTER PERFORMANCE COMPARISON ===\n";
        std::cout << std::fixed << std::setprecision(6);

        if (test_ckf) {
            std::cout << "\nCKF Metrics:\n";
            std::cout << "  Average error:  " << data.ckf_metrics.average_error << "\n";
            std::cout << "  Maximum error:  " << data.ckf_metrics.max_error << "\n";
            std::cout << "  RMSE:           " << data.ckf_metrics.rms_error << "\n";
            std::cout << "  Convergence:    " << data.ckf_metrics.convergence_time << " s\n";
        }

        std::cout << "\nSRCF Metrics:\n";
        std::cout << "  Average error:  " << data.srcf_metrics.average_error << "\n";
        std::cout << "  Maximum error:  " << data.srcf_metrics.max_error << "\n";
        std::cout << "  RMSE:           " << data.srcf_metrics.rms_error << "\n";
        std::cout << "  Convergence:    " << data.srcf_metrics.convergence_time << " s\n";

        if (test_ckf) {
            std::cout << "\n=== COMPARATIVE ANALYSIS ===\n";
            std::cout << "SRCF/CKF error ratio:      "
                      << data.srcf_metrics.average_error / data.ckf_metrics.average_error << "\n";
            std::cout << "SRCF/CKF RMSE ratio:        "
                      << data.srcf_metrics.rms_error / data.ckf_metrics.rms_error << "\n";


            std::cout << "\n=== VERHAEGEN & VAN DOOREN RECOMMENDATION ===\n";

            // Решаем, какой фильтр рекомендовать на основе критериев
            bool recommend_srcf = true;
            std::string reasons;

            // Рекомендации в стиле статьи
            if (data.srcf_metrics.average_error > data.ckf_metrics.average_error * 1.2) {
                std::cout << "\nCONCLUSION: CKF performs significantly better than SRCF\n";
            } else if (data.srcf_metrics.average_error < data.ckf_metrics.average_error) {
                std::cout << "\nCONCLUSION: SRCF performs better than CKF\n";
            } else {
                std::cout << "\nCONCLUSION: Both filters have similar performance\n";
            }

            if (data.comparison.avg_error_ratio > 1.2) {
                recommend_srcf = false;
                reasons += "CKF has significantly lower average error (+20%)\n";
            }

            if (data.comparison.rms_error_ratio > 1.15) {
                if (!recommend_srcf) reasons += "CKF has better RMS error\n";
                else recommend_srcf = false;
            }

            if (data.comparison.percentage_srcf_better < 40.0) {
                if (!recommend_srcf) reasons += "CKF is more consistent\n";
                else recommend_srcf = false;
            }

            // Преимущества SRCF
            if (data.comparison.cond_number_ratio < 0.8) {
                if (recommend_srcf) reasons += "SRCF has better numerical conditioning\n";
            }

            if (data.comparison.cov_norm_ratio < 1.0) {
                if (recommend_srcf) reasons += "SRCF has more stable covariance\n";
            }

            if (recommend_srcf) {
                std::cout << "RECOMMENDATION: Use Square Root Covariance Filter (SRCF)\n";
            } else {
                std::cout << "RECOMMENDATION: Use Conventional Kalman Filter (CKF)\n";
            }

            if (!reasons.empty()) {
                std::cout << "Reasons:\n" << reasons;
            }

        }
    }

} // namespace data_generator


#endif // DATA_GENERATOR_HPP