// data_generator.hpp
#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <memory>
#include "kalman.hpp"
#include "models.hpp"
#include "time_generator.hpp"

namespace data_generator {

    enum class DataFormat {
        BINARY,      // Бинарный формат для скорости
        TEXT_CSV,    // Текстовый CSV для удобства
        TEXT_MATLAB, // Текстовый формат MATLAB
        TEXT_TXT     // Простой TXT формат
    };

    struct SimulationConfig {
        size_t total_steps = 1000;
        double base_dt = 0.01;
        bool add_process_noise = true;
        bool add_measurement_noise = true;
        double process_noise_scale = 1.0;
        double measurement_noise_scale = 1.0;
        model2::ControlScenario scenario = model2::ControlScenario::SINE_WAVE;
        time_generator::TimeMode time_mode = time_generator::TimeMode::RANDOM_JITTER;
        DataFormat format = DataFormat::BINARY;
        std::string output_dir = "./data";
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
    };

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
                Eigen::VectorXd u = model2::u(t, config_.scenario);

                // Шумы
                // Генератор данных // Шумы
                Eigen::VectorXd w;
                if (config_.add_process_noise) {
                    w = model2::w(t, dt, true) * config_.process_noise_scale;
                } else {
                    w = Eigen::Vector2d::Zero();
                }

                Eigen::VectorXd v;
                if (config_.add_measurement_noise) {
                    v = model2::v(t, true) * config_.measurement_noise_scale;
                } else {
                    v = Eigen::Vector2d::Zero();
                }

                // Динамика системы
                x_true = A * x_true + B * u + w;

                // Измерения
                Eigen::Vector2d y_exact = C * x_true;
                Eigen::Vector2d y_noisy = y_exact + v;

                // Шаги фильтров
                ckf.step(A, B, C, D, Q, R, u, y_noisy);
                srcf.step(A, B, C, D, Q, R, u, y_noisy);

                // Сохранение данных
                data.true_states.push_back(x_true);
                data.measurements.push_back(y_exact);
                data.noisy_measurements.push_back(y_noisy);
                data.controls.push_back(u);
                data.ckf_estimates.push_back(ckf.state());
                data.srcf_estimates.push_back(srcf.state());
                data.ckf_covariances.push_back(ckf.covariance());
                data.srcf_covariances.push_back(srcf.covariance());
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
        void saveBinary(const SimulationData& data) {
            std::string prefix = config_.output_dir + "/";

            // Сохранение времен
            saveVectorBinary(data.times, prefix + "times.bin");

            // Сохранение векторов состояния
            saveVector2dBinary(data.true_states, prefix + "true_states.bin");
            saveVector2dBinary(data.measurements, prefix + "measurements.bin");
            saveVector2dBinary(data.noisy_measurements, prefix + "noisy_measurements.bin");
            saveVector2dBinary(data.ckf_estimates, prefix + "ckf_estimates.bin");
            saveVector2dBinary(data.srcf_estimates, prefix + "srcf_estimates.bin");

            // Сохранение управлений (скаляр)
            std::vector<double> controls_scalar;
            for (const auto& u : data.controls) {
                controls_scalar.push_back(u(0));
            }
            saveVectorBinary(controls_scalar, prefix + "controls.bin");

            // Сохранение ковариаций
            saveMatrix2dBinary(data.ckf_covariances, prefix + "ckf_covariances.bin");
            saveMatrix2dBinary(data.srcf_covariances, prefix + "srcf_covariances.bin");
        }

        void saveTextCSV(const SimulationData& data) {
            std::string filename = config_.output_dir + "/simulation_data.csv";
            std::ofstream file(filename);

            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            // Заголовок
            file << "time,true_phi,true_p,meas_phi_exact,meas_p_exact,"
                 << "meas_phi_noisy,meas_p_noisy,control,"
                 << "ckf_phi,ckf_p,srcf_phi,srcf_p,"
                 << "ckf_cov_11,ckf_cov_12,ckf_cov_21,ckf_cov_22,"
                 << "srcf_cov_11,srcf_cov_12,srcf_cov_21,srcf_cov_22\n";

            file << std::fixed << std::setprecision(12);

            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << data.times[i] << ","
                     << data.true_states[i](0) << "," << data.true_states[i](1) << ","
                     << data.measurements[i](0) << "," << data.measurements[i](1) << ","
                     << data.noisy_measurements[i](0) << "," << data.noisy_measurements[i](1) << ","
                     << data.controls[i](0) << ","
                     << data.ckf_estimates[i](0) << "," << data.ckf_estimates[i](1) << ","
                     << data.srcf_estimates[i](0) << "," << data.srcf_estimates[i](1) << ","
                     << data.ckf_covariances[i](0,0) << "," << data.ckf_covariances[i](0,1) << ","
                     << data.ckf_covariances[i](1,0) << "," << data.ckf_covariances[i](1,1) << ","
                     << data.srcf_covariances[i](0,0) << "," << data.srcf_covariances[i](0,1) << ","
                     << data.srcf_covariances[i](1,0) << "," << data.srcf_covariances[i](1,1) << "\n";
            }

            file.close();

            // Сохранение конфигурации
            saveConfig();
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
            file << "ckf_estimates = [";
            for (size_t i = 0; i < data.ckf_estimates.size(); ++i) {
                file << data.ckf_estimates[i](0) << ", " << data.ckf_estimates[i](1);
                if (i < data.ckf_estimates.size() - 1) file << "; ";
            }
            file << "];\n\n";

            file << "srcf_estimates = [";
            for (size_t i = 0; i < data.srcf_estimates.size(); ++i) {
                file << data.srcf_estimates[i](0) << ", " << data.srcf_estimates[i](1);
                if (i < data.srcf_estimates.size() - 1) file << "; ";
            }
            file << "];\n";

            file.close();

            // Сохранение конфигурации
            saveConfig();
        }

        void saveConfig() {
            std::string filename = config_.output_dir + "/config.txt";
            std::ofstream file(filename);

            file << "=== Simulation Configuration ===\n\n";
            file << "Total steps: " << config_.total_steps << "\n";
            file << "Base dt: " << config_.base_dt << " s\n";
            file << "Add process noise: " << (config_.add_process_noise ? "YES" : "NO") << "\n";
            file << "Add measurement noise: " << (config_.add_measurement_noise ? "YES" : "NO") << "\n";
            file << "Process noise scale: " << config_.process_noise_scale << "\n";
            file << "Measurement noise scale: " << config_.measurement_noise_scale << "\n";

            file << "Control scenario: ";
            switch (config_.scenario) {
                case model2::ControlScenario::ZERO_HOLD: file << "ZERO_HOLD"; break;
                case model2::ControlScenario::STEP_MANEUVER: file << "STEP_MANEUVER"; break;
                case model2::ControlScenario::SINE_WAVE: file << "SINE_WAVE"; break;
                case model2::ControlScenario::PULSE: file << "PULSE"; break;
            }
            file << "\n";

            file << "Time mode: ";
            switch (config_.time_mode) {
                case time_generator::TimeMode::UNIFORM: file << "UNIFORM"; break;
                case time_generator::TimeMode::VARIABLE: file << "VARIABLE"; break;
                case time_generator::TimeMode::RANDOM_JITTER: file << "RANDOM_JITTER"; break;
            }
            file << "\n";

            file << "Data format: ";
            switch (config_.format) {
                case DataFormat::BINARY: file << "BINARY"; break;
                case DataFormat::TEXT_CSV: file << "TEXT_CSV"; break;
                case DataFormat::TEXT_MATLAB: file << "TEXT_MATLAB"; break;
            }
            file << "\n";

            file.close();
        }

        // Вспомогательные функции для бинарного сохранения
        void saveVectorBinary(const std::vector<double>& vec, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = vec.size();
            file.write((char*)&n, sizeof(size_t));
            file.write((char*)vec.data(), n * sizeof(double));
            file.close();
        }

        void saveVector2dBinary(const std::vector<Eigen::Vector2d>& vec, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = vec.size();
            file.write((char*)&n, sizeof(size_t));
            for (const auto& v : vec) {
                file.write((char*)v.data(), 2 * sizeof(double));
            }
            file.close();
        }

        void saveMatrix2dBinary(const std::vector<Eigen::Matrix2d>& mats, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            size_t n = mats.size();
            file.write((char*)&n, sizeof(size_t));
            for (const auto& m : mats) {
                file.write((char*)m.data(), 4 * sizeof(double));
            }
            file.close();
        }
        void saveTextTXT(const SimulationData& data) {
            // Сохраняем данные в нескольких TXT файлах
            std::string prefix = config_.output_dir + "/";

            // 1. Основной файл с состояниями и оценками
            saveMainDataTXT(data, prefix + "main_data.txt");

            // 2. Файл с ковариациями
            saveCovarianceTXT(data, prefix + "covariances.txt");

            // 3. Файл с измерениями
            saveMeasurementsTXT(data, prefix + "measurements.txt");

            // 4. Файл со статистикой
            saveStatisticsTXT(data, prefix + "statistics.txt");

            // 5. Конфигурация
            saveConfig();
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
                 << std::setw(15) << "CKF_phi"
                 << std::setw(15) << "CKF_p"
                 << std::setw(15) << "SRCF_phi"
                 << std::setw(15) << "SRCF_p"
                 << std::setw(12) << "Control"
                 << "\n";

            file << std::string(120, '-') << "\n";

            for (size_t i = 0; i < data.times.size() - 1; ++i) {
                file << std::setw(12) << data.times[i]
                     << std::setw(15) << data.true_states[i](0)
                     << std::setw(15) << data.true_states[i](1)
                     << std::setw(15) << data.ckf_estimates[i](0)
                     << std::setw(15) << data.ckf_estimates[i](1)
                     << std::setw(15) << data.srcf_estimates[i](0)
                     << std::setw(15) << data.srcf_estimates[i](1)
                     << std::setw(12) << data.controls[i](0)
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
                    file << "CKF Covariance:\n";
                    file << "  [" << data.ckf_covariances[i](0,0) << ", "
                         << data.ckf_covariances[i](0,1) << "]\n";
                    file << "  [" << data.ckf_covariances[i](1,0) << ", "
                         << data.ckf_covariances[i](1,1) << "]\n";
                    file << "  Determinant: " << data.ckf_covariances[i].determinant() << "\n";

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

            // Статистика по шагам
            file << "Error Statistics by Steps:\n";
            file << std::setw(8) << "Step"
                 << std::setw(15) << "CKF_Error"
                 << std::setw(15) << "SRCF_Error"
                 << std::setw(15) << "Difference"
                 << "\n";

            file << std::string(60, '-') << "\n";

            for (size_t i = 0; i < std::min((size_t)50, ckf_errors.size()); i += 5) {
                file << std::setw(8) << i
                     << std::setw(15) << ckf_errors[i]
                     << std::setw(15) << srcf_errors[i]
                     << std::setw(15) << (srcf_errors[i] - ckf_errors[i])
                     << "\n";
            }

            file.close();
        }
    };

} // namespace data_generator

#endif // DATA_GENERATOR_HPP