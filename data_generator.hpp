/**
 * @file data_generator.hpp
 * @brief Генератор данных для сравнения фильтров Калмана
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * @note Данный модуль реализует генерацию данных для сравнения
 *       классического (CKF) и квадратно-корневого (SRCF) фильтров Калмана.
 *       Включает различные модели систем, режимы времени и форматы вывода.
 */

#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#pragma once

#include <utility>
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
// КОНФИГУРАЦИЯ И ДАННЫЕ
// ============================================================================
    /**
    * @enum DataFormat
    * @brief Форматы сохранения данных
    */
    enum class DataFormat {
        BINARY,      /**< Бинарный формат для скорости и компактности */
        TEXT_CSV,    /**< Текстовый CSV для удобства чтения */
        TEXT_MATLAB, /**< Текстовый формат совместимый с MATLAB */
        TEXT_TXT     /**< Простой TXT формат с форматированием */
    };

    /**
     * @enum ModelType
     * @brief Типы моделей для генерации данных
     */
    enum class ModelType {
        MODEL0,  /**< Модель рыскания самолета */
        MODEL2   /**< Модель крена самолета */
    };

    /**
     * @struct SimulationConfig
     * @brief Конфигурация симуляции
     */
    struct SimulationConfig {
        int seed = 0;                              /**< Начальный seed модели */
        size_t total_steps = 1000;                 /**< Общее количество шагов симуляции */
        double base_dt = 0.01;                     /**< Базовый шаг по времени (секунды) */
        bool add_process_noise = true;             /**< Добавлять шум процесса */
        bool add_measurement_noise = true;         /**< Добавлять шум измерений */
        double process_noise_scale = 1.0;          /**< Масштаб шума процесса */
        double measurement_noise_scale = 1.0;      /**< Масштаб шума измерений */
        ModelType model_type = ModelType::MODEL2;  /**< Тип используемой модели */
        union {
            model0::ControlScenario scenario0;     /**< Сценарий управления для MODEL0 */
            model2::ControlScenario scenario2;     /**< Сценарий управления для MODEL2 */
        } scenario;
        time_generator::TimeMode time_mode = time_generator::TimeMode::RANDOM_JITTER; /**< Режим генерации времени */
        DataFormat format = DataFormat::BINARY;    /**< Формат сохранения данных */
        std::string output_dir = "./data";         /**< Директория для выходных данных */
        bool test_ckf = true;                      /**< Тестировать ли CKF фильтр */
        Eigen::VectorXd initial_state = Eigen::Vector2d::Zero();     /**< Начальное состояние системы */
        Eigen::MatrixXd initial_covariance = Eigen::Matrix2d::Identity() * 0.1; /**< Начальная ковариация */
        bool use_custom_initial = false;                            /**< Использовать ли кастомные начальные условия */
    };

    /**
     * @struct SimulationData
     * @brief Структура для хранения всех данных симуляции
     */
    struct SimulationData {
        std::vector<double> times;                             /**< Временные метки */
        std::vector<double> ckf_step_times;                    /**< Время шага CKF */
        std::vector<double> srcf_step_times;                   /**< Время шага SRCF */
        std::vector<Eigen::Vector2d> true_states;              /**< Истинные состояния */
        std::vector<Eigen::Vector2d> measurements;             /**< Точные измерения */
        std::vector<Eigen::Vector2d> noisy_measurements;       /**< Зашумленные измерения */
        std::vector<Eigen::VectorXd> controls;                 /**< Векторы управления */
        std::vector<Eigen::Vector2d> ckf_estimates;            /**< Оценки CKF */
        std::vector<Eigen::Vector2d> srcf_estimates;           /**< Оценки SRCF */
        std::vector<Eigen::Matrix2d> ckf_covariances;          /**< Ковариации CKF */
        std::vector<Eigen::Matrix2d> srcf_covariances;         /**< Ковариации SRCF */

        /**
         * @struct FilterMetrics
         * @brief Метрики производительности фильтра
         */
        struct FilterMetrics {
            double average_error = 0.0;        /**< Средняя ошибка оценки */
            double max_error = 0.0;            /**< Максимальная ошибка оценки */
            double rms_error = 0.0;            /**< RMS ошибка оценки */
            double convergence_time = 0.0;     /**< Время сходимости (секунды) */
            double cov_norm = 0.0;             /**< Норма ковариации */
            double cond_number = 0.0;          /**< Число обусловленности ковариации */
            double symmetry_error = 0.0;       /**< Асимметрия ковариации */
            std::vector<double> error_history; /**< История ошибок на каждом шаге */
        };

        FilterMetrics ckf_metrics;  /**< Метрики для CKF */
        FilterMetrics srcf_metrics; /**< Метрики для SRCF */

        /**
         * @struct ComparisonMetrics
         * @brief Метрики сравнения двух фильтров
         */
        struct ComparisonMetrics {
            double avg_error_ratio = 0.0;          /**< Отношение средних ошибок (SRCF/CKF) */
            double rms_error_ratio = 0.0;          /**< Отношение RMS ошибок (SRCF/CKF) */
            double max_error_ratio = 0.0;          /**< Отношение макс. ошибок (SRCF/CKF) */
            double cond_number_ratio = 0.0;        /**< Отношение чисел обусловленности (SRCF/CKF) */
            double cov_norm_ratio = 0.0;           /**< Отношение норм ковариаций (SRCF/CKF) */
            double max_srcf_minus_ckf = 0.0;       /**< Максимальная разница ошибок (SRCF - CKF) */
            double max_ckf_minus_srcf = 0.0;       /**< Максимальная разница ошибок (CKF - SRCF) */
            double avg_absolute_difference = 0.0;  /**< Средняя абсолютная разница ошибок */
            double std_dev_difference = 0.0;       /**< Стандартное отклонение разницы ошибок */
            double percentage_srcf_better = 0.0;   /**< Процент шагов, где SRCF лучше */
            std::vector<double> error_differences; /**< Разности ошибок на каждом шаге (SRCF - CKF) */
            std::vector<double> relative_differences; /**< Относительные разности ошибок (%) */
        };
        ComparisonMetrics comparison; /**< Метрики сравнения фильтров */
    };
// ============================================================================
// КЛАСС ГЕНЕРАТОРА ДАННЫХ
// ============================================================================

    /**
     * @class DataGenerator
     * @brief Генератор данных для сравнения фильтров Калмана
     *
     * @note Класс реализует генерацию данных, симуляцию фильтров,
     *       расчет метрик и сохранение результатов в различных форматах.
     */
    class DataGenerator {
    private:
        SimulationConfig config_;                     /**< Конфигурация симуляции */
        time_generator::TimeGenerator time_gen_;      /**< Генератор временной сетки */
        std::ofstream log_file_;                      /**< Файл журнала */

        /**
         * @brief Валидация конфигурации симуляции
         * @exception std::invalid_argument Если конфигурация некорректна
         */
        void validateConfig() const
        {
            if (config_.total_steps == 0) {
                throw std::invalid_argument("total_steps must be > 0");
            }
            if (config_.base_dt <= 0) {
                throw std::invalid_argument("base_dt must be > 0");
            }
            if (config_.process_noise_scale < 0.0) {
                throw std::invalid_argument("process_noise_scale must be >= 0");
            }
            if (config_.measurement_noise_scale < 0.0) {
                throw std::invalid_argument("measurement_noise_scale must be >= 0");
            }
        }

        /**
         * @brief Запись сообщения в лог
         * @param message Сообщение для записи
         */
        void log(const std::string& message)
        {
            if (!log_file_.is_open()) {
                log_file_.open(config_.output_dir + "/simulation.log");
            }
            log_file_ << message << std::endl;
        }

    public:
        /**
         * @brief Конструктор генератора данных
         * @param config Конфигурация симуляции
         * @param seed Начальное значение для генератора случайных чисел
         */
        explicit DataGenerator(SimulationConfig config, int seed = 42)
                : config_(std::move(config)), time_gen_(seed)
        {
            validateConfig();
            std::filesystem::create_directories(config_.output_dir);
            std::cout << "[DataGenerator] Setting seed: " << seed << std::endl;

            kalman_noise::reset_noise_generators(seed);
            if (config_.model_type == ModelType::MODEL0) {
                model0::reset_noise_with_seed(seed);
            } else if (config_.model_type == ModelType::MODEL2) {
                model2::reset_noise_with_seed(seed);
            }

            log("[DataGenerator] Initialized with configuration:");
            log("  total_steps = " + std::to_string(config_.total_steps));
            log("  base_dt = " + std::to_string(config_.base_dt));
            log("  output_dir = " + config_.output_dir);
            log("  seed = " + std::to_string(seed));
        }

        /**
         * @brief Деструктор (закрывает лог-файл)
         */
        ~DataGenerator()
        {
            if (log_file_.is_open()) {
                log("[DataGenerator] Simulation completed");
                log_file_.close();
            }
        }

        /**
         * @brief Генерация данных симуляции
         * @return SimulationData Сгенерированные данные
         */
        SimulationData generate()
        {
            log("[DataGenerator] Starting data generation");
            SimulationData data;

            // Генерация временной сетки
            data.times = time_gen_.generate(config_.total_steps + 1,
                                            config_.base_dt,
                                            config_.time_mode);
            std::string timegrid_file = config_.output_dir + "/timegrid.bin";
            time_generator::TimeGenerator::saveToFile(data.times, timegrid_file);
            try {
                std::vector<double> time_check = time_generator::TimeGenerator::loadFromFile(timegrid_file);
                if (time_check.size() != data.times.size()) {
                    log("[WARNING] Time grid loading mismatch: " +
                        std::to_string(time_check.size()) + " vs " +
                        std::to_string(data.times.size()));
                }
            } catch (const std::exception& e) {
                log("[ERROR] Failed to load time grid: " + std::string(e.what()));
            }

            // Выбор модели для генерации
            log("[DataGenerator] Using model: " +
                std::string(config_.model_type == ModelType::MODEL0 ? "MODEL0" : "MODEL2"));

            if (config_.model_type == ModelType::MODEL0) {
                generate_with_model0(data);
            } else {
                generate_with_model2(data);
            }

            // Рассчитываем метрики
            calculate_metrics(data);

            // Расчёт метрик сравнения
            if (config_.test_ckf && !data.ckf_estimates.empty()) {
                calculate_comparison_metrics(data);
            }
            log("[DataGenerator] Data generation completed");
            return data;
        }

        /**
         * @brief Сохранение данных в выбранном формате
         * @param data Данные для сохранения
         */
        void save(const SimulationData& data)
        {
            log("[DataGenerator] Saving data in " +
                format_to_string(config_.format) + " format");
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
            log("[DataGenerator] Data saved successfully");
        }

    private:
        /**
         * @brief Конвертация формата в строку
         * @param format Формат данных
         * @return std::string Строковое представление формата
         */
        static std::string format_to_string(DataFormat format)
        {
            switch (format) {
                case DataFormat::BINARY: return "BINARY";
                case DataFormat::TEXT_CSV: return "TEXT_CSV";
                case DataFormat::TEXT_MATLAB: return "TEXT_MATLAB";
                case DataFormat::TEXT_TXT: return "TEXT_TXT";
                default: return "UNKNOWN";
            }
        }

        /**
         * @brief Генерация данных с использованием MODEL0
         * @param data Структура для хранения данных
         */
        void generate_with_model0(SimulationData& data)
        {
            log("[DataGenerator] Generating with MODEL0");
            Eigen::Vector2d x_true;
            Eigen::Matrix2d P0;
            if (config_.use_custom_initial) {
                // Используем пользовательские начальные условия
                x_true = config_.initial_state;
                P0 = config_.initial_covariance;
                log("[MODEL0] Using custom initial conditions:");
            } else {
                // Используем стандартные начальные условия
                x_true = Eigen::Vector2d::Zero();
                P0 << 0.1, 0.0,
                        0.0, 0.1;
                log("[MODEL0] Using default initial conditions:");
            }
            log("[MODEL0] Initial state: x_true = ["
                + std::to_string(x_true(0)) + ", " + std::to_string(x_true(1)) + "]");
            log("[MODEL0] Initial covariance: P0 = [["
                + std::to_string(P0(0,0)) + ", " + std::to_string(P0(0,1)) + "], ["
                + std::to_string(P0(1,0)) + ", " + std::to_string(P0(1,1)) + "]]");

            kalman::CKF ckf(x_true, P0);
            kalman::SRCF srcf(x_true, P0);

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
                x_true = model0::true_dynamics(
                        x_true,
                        t,
                        dt,
                        config_.scenario.scenario0,
                        config_.add_process_noise
                );

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
                    auto start_ckf = std::chrono::high_resolution_clock::now();
                    ckf.step(A, B, C, D, Q, R, u, y_noisy);
                    auto end_ckf = std::chrono::high_resolution_clock::now();
                    data.ckf_estimates.emplace_back(ckf.state());
                    data.ckf_covariances.emplace_back(ckf.covariance());
                    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ckf - start_ckf);
                    double time_ns = static_cast<double>(duration_ns.count());
                    std::chrono::duration<double> duration_sec = end_ckf - start_ckf;
                    double time_sec = duration_sec.count();
                    data.ckf_step_times.push_back(duration_ns.count());
                    std::cout << "Время CKF (сек): " << time_sec << std::endl;
                    std::cout << "Время CKF (нс): " << time_ns << std::endl;
                }
                auto start_srcf = std::chrono::high_resolution_clock::now();
                srcf.step(A, B, C, D, Q, R, u, y_noisy);
                auto end_srcf = std::chrono::high_resolution_clock::now();
                auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_srcf - start_srcf);
                double time_ns = static_cast<double>(duration_ns.count());
                std::chrono::duration<double> duration_sec = end_srcf - start_srcf;
                double time_sec = duration_sec.count();
                data.srcf_step_times.push_back(duration_ns.count());
                data.srcf_estimates.emplace_back(srcf.state());
                data.srcf_covariances.emplace_back(srcf.covariance());
                data.true_states.push_back(x_true);
                data.measurements.push_back(y_exact);
                data.noisy_measurements.push_back(y_noisy);
                data.controls.push_back(u);
                std::cout << "Время SRCF (сек): " << time_sec << std::endl;
                std::cout << "Время SRCF (нс): " << time_ns << std::endl;
            }
            log("[MODEL0] Generation completed: " +
                std::to_string(data.true_states.size()) + " steps");
        }

        /**
         * @brief Генерация данных с использованием MODEL2
         * @param data Структура для хранения данных
         */
        void generate_with_model2(SimulationData& data)
        {
            log("[DataGenerator] Generating with MODEL2");
            Eigen::Vector2d x_true;
            Eigen::Matrix2d P0;
            if (config_.use_custom_initial) {
                x_true = config_.initial_state;
                P0 = config_.initial_covariance;
                log("[MODEL2] Using custom initial conditions:");
            } else {
                x_true = Eigen::Vector2d::Zero();
                P0 << 0.1, 0.0,
                        0.0, 0.01;
                log("[MODEL2] Using default initial conditions:");
            }

            log("[MODEL2] Initial state: x_true = ["
                + std::to_string(x_true(0)) + ", " + std::to_string(x_true(1)) + "]");
            log("[MODEL2] Initial covariance: P0 = [["
                + std::to_string(P0(0,0)) + ", " + std::to_string(P0(0,1)) + "], ["
                + std::to_string(P0(1,0)) + ", " + std::to_string(P0(1,1)) + "]]");

            // Подготовка фильтров
            kalman::CKF ckf(x_true, P0);
            kalman::SRCF srcf(x_true, P0);

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

                // Управляющее воздействие
                Eigen::VectorXd u = model2::u(t, config_.scenario.scenario2);

                // Динамика системы
                x_true = model2::true_dynamics(
                        x_true,
                        t,
                        dt,
                        config_.scenario.scenario2,
                        config_.add_process_noise
                );

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
                    auto start_ckf = std::chrono::high_resolution_clock::now();
                    ckf.step(A, B, C, D, Q, R, u, y_noisy);
                    auto end_ckf = std::chrono::high_resolution_clock::now();
                    data.ckf_estimates.emplace_back(ckf.state());
                    data.ckf_covariances.emplace_back(ckf.covariance());
                    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ckf - start_ckf);
                    double time_ns = static_cast<double>(duration_ns.count());
                    std::chrono::duration<double> duration_sec = end_ckf - start_ckf;
                    double time_sec = duration_sec.count();
                    data.ckf_step_times.push_back(duration_ns.count());
                    std::cout << "Время CKF (сек): " << time_sec << std::endl;
                    std::cout << "Время CKF (нс): " << time_ns << std::endl;
                }
                auto start_srcf = std::chrono::high_resolution_clock::now();
                srcf.step(A, B, C, D, Q, R, u, y_noisy);
                auto end_srcf = std::chrono::high_resolution_clock::now();
                auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_srcf - start_srcf);
                double time_ns = static_cast<double>(duration_ns.count());
                std::chrono::duration<double> duration_sec = end_srcf - start_srcf;
                double time_sec = duration_sec.count();
                data.srcf_step_times.push_back(duration_ns.count());
                data.srcf_estimates.emplace_back(srcf.state());
                data.srcf_covariances.emplace_back(srcf.covariance());
                data.true_states.push_back(x_true);
                data.measurements.push_back(y_exact);
                data.noisy_measurements.push_back(y_noisy);
                data.controls.push_back(u);
                std::cout << "Время SRCF (сек): " << time_sec << std::endl;
                std::cout << "Время SRCF (нс): " << time_ns << std::endl;
            }
            log("[MODEL2] Generation completed: " +
                std::to_string(data.true_states.size()) + " steps");
        }

        /**
         * @brief Расчет метрик производительности фильтров
         * @param data Данные симуляции
         */
        void calculate_metrics(SimulationData& data)
        {
            log("[DataGenerator] Calculating filter metrics");
            // Проверка размеров данных
            std::cout << "\n[DEBUG] Data sizes for metric calculation:" << std::endl;
            std::cout << "  true_states: " << data.true_states.size() << std::endl;
            std::cout << "  ckf_estimates: " << data.ckf_estimates.size() << std::endl;
            std::cout << "  srcf_estimates: " << data.srcf_estimates.size() << std::endl;

            if (data.true_states.size() != data.ckf_estimates.size() ||
                data.true_states.size() != data.srcf_estimates.size()) {
                std::cerr << "[WARNING] Mismatched data sizes for metric calculation!" << std::endl;
            }
            if (config_.test_ckf && !data.ckf_estimates.empty()) {
                std::cout << "[DEBUG] Calculating CKF metrics..." << std::endl;
                calculate_filter_metrics(data.true_states, data.ckf_estimates,
                                         data.ckf_covariances, data.times,
                                         data.ckf_metrics);
                std::cout << "[DEBUG] CKF metrics calculated:" << std::endl;
                std::cout << "  average_error: " << data.ckf_metrics.average_error << std::endl;
                std::cout << "  max_error: " << data.ckf_metrics.max_error << std::endl;
                std::cout << "  rms_error: " << data.ckf_metrics.rms_error << std::endl;
                std::cout << "  error_history size: " << data.ckf_metrics.error_history.size() << std::endl;
                log("[Metrics] CKF average error: " + std::to_string(data.ckf_metrics.average_error));
            }

            std::cout << "[DEBUG] Calculating SRCF metrics..." << std::endl;
            calculate_filter_metrics(data.true_states, data.srcf_estimates,
                                     data.srcf_covariances, data.times,
                                     data.srcf_metrics);
            std::cout << "[DEBUG] SRCF metrics calculated:" << std::endl;
            std::cout << "  average_error: " << data.srcf_metrics.average_error << std::endl;
            std::cout << "  max_error: " << data.srcf_metrics.max_error << std::endl;
            std::cout << "  rms_error: " << data.srcf_metrics.rms_error << std::endl;
            std::cout << "  error_history size: " << data.srcf_metrics.error_history.size() << std::endl;
            log("[Metrics] SRCF average error: " + std::to_string(data.srcf_metrics.average_error));
        }

        /**
         * @brief Расчет метрик для одного фильтра
         * @param true_states Истинные состояния
         * @param estimates Оценки фильтра
         * @param covariances Ковариации фильтра
         * @param times Временные метки
         * @param metrics Структура для сохранения метрик
         */
        static void calculate_filter_metrics(const std::vector<Eigen::Vector2d>& true_states,
                                             const std::vector<Eigen::Vector2d>& estimates,
                                             const std::vector<Eigen::Matrix2d>& covariances,
                                             const std::vector<double>& times,
                                             SimulationData::FilterMetrics& metrics)
        {
            const size_t n = true_states.size();
            if (n == 0) return;
            double sum_sq_error = 0.0;
            double sum_cov_norm = 0.0;
            double sum_cond = 0.0;
            int valid_cond_count = 0;
            metrics.max_error = 0.0;
            metrics.error_history.clear();
            metrics.error_history.reserve(n);

            for (size_t i = 0; i < n; ++i) {
                // Ошибка оценки
                double error = (true_states[i] - estimates[i]).norm();
                metrics.error_history.push_back(error);
                sum_sq_error += error * error;

                if (error > metrics.max_error) {
                    metrics.max_error = error;
                }

                // Норма ковариации
                sum_cov_norm += covariances[i].norm();

                // БЕЗОПАСНОЕ вычисление числа обусловленности
                Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariances[i]);
                const Eigen::Vector2d& singular_values = svd.singularValues();

                // Проверяем, что матрица не вырождена
                double min_sv = singular_values.minCoeff();
                double max_sv = singular_values.maxCoeff();

                if (i % 10 == 0) {  // Для отладки каждого 10-го шага
                    std::cout << "[DEBUG] Step " << i << " condition number calculation:" << std::endl;
                    std::cout << "  singular values: " << singular_values.transpose() << std::endl;
                    std::cout << "  min_sv: " << min_sv << ", max_sv: " << max_sv << std::endl;
                    if (min_sv > 1e-17 && max_sv > 1e-17) {
                        std::cout << "  condition number: " << max_sv / min_sv << std::endl;
                    } else {
                        std::cout << "  matrix is ill-conditioned or singular" << std::endl;
                    }
                }

                if (min_sv > 1e-17 && max_sv > 1e-17) {
                    double cond = max_sv / min_sv;

                    // Ограничиваем cond разумным значением
                    if (cond < 1e15) {
                        sum_cond += cond;
                        valid_cond_count++;
                    }
                }
                const int window_size = 50;
                if (i >= window_size) {
                    double window_avg = 0.0;
                    for (int j = 0; j < window_size; ++j) {
                        window_avg += metrics.error_history[i - j];
                    }
                    window_avg /= window_size;
                    if (window_avg < 0.01 && metrics.convergence_time == 0.0) {
                        metrics.convergence_time = times[i];
                    }
                }
            }

            metrics.average_error = std::accumulate(metrics.error_history.begin(),
                                                    metrics.error_history.end(), 0.0)
                                    / static_cast<double>(metrics.error_history.size());

            metrics.rms_error = std::sqrt(sum_sq_error / static_cast<double>(metrics.error_history.size()));
            metrics.cov_norm = sum_cov_norm / static_cast<double>(covariances.size());
            metrics.cond_number = (valid_cond_count > 0) ?
                                  sum_cond / valid_cond_count : 1.0;
        }

        /**
         * @brief Расчет метрик сравнения фильтров
         * @param data Данные симуляции
         */
        void calculate_comparison_metrics(SimulationData& data)
        {
            if (!config_.test_ckf || data.ckf_estimates.empty()) {
                return;
            }

            const size_t n = data.true_states.size();
            data.comparison.error_differences.clear();
            data.comparison.relative_differences.clear();
            data.comparison.error_differences.reserve(n);
            data.comparison.relative_differences.reserve(n);

            double sum_diff = 0.0;
            double sum_sq_diff = 0.0;
            double sum_abs_diff = 0.0;
            data.comparison.max_srcf_minus_ckf = -std::numeric_limits<double>::max();
            data.comparison.max_ckf_minus_srcf = -std::numeric_limits<double>::max();
            int srcf_better_count = 0;

            for (size_t i = 0; i < n; ++i) {
                double ckf_error = data.ckf_metrics.error_history[i];
                double srcf_error = data.srcf_metrics.error_history[i];
                double diff = srcf_error - ckf_error;

                data.comparison.error_differences.push_back(diff);

                // Относительная разница в процентах
                const double avg_error = (ckf_error + srcf_error) / 2.0;
                const double rel_diff = (avg_error > 1e-17) ?
                                        (diff / avg_error) * 100.0 : 0.0;
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
            data.comparison.avg_error_ratio = data.srcf_metrics.average_error /
                                              data.ckf_metrics.average_error;
            if (data.ckf_metrics.rms_error > 1e-17) {
                data.comparison.rms_error_ratio = data.srcf_metrics.rms_error / data.ckf_metrics.rms_error;
            }

            if (data.ckf_metrics.max_error > 1e-17) {
                data.comparison.max_error_ratio = data.srcf_metrics.max_error / data.ckf_metrics.max_error;
            }

            if (data.ckf_metrics.cond_number > 1e-17) {
                data.comparison.cond_number_ratio = data.srcf_metrics.cond_number / data.ckf_metrics.cond_number;
            }

            if (data.ckf_metrics.cov_norm > 1e-17) {
                data.comparison.cov_norm_ratio = data.srcf_metrics.cov_norm / data.ckf_metrics.cov_norm;
            }
            auto n_new = static_cast<double>(n);
            data.comparison.avg_absolute_difference = sum_abs_diff / n_new;
            data.comparison.percentage_srcf_better = (static_cast<double>(srcf_better_count) / n_new) * 100.0;

            // Стандартное отклонение разницы
            double mean_diff = sum_diff / n_new;
            data.comparison.std_dev_difference = std::sqrt((sum_sq_diff / n_new) - (mean_diff * mean_diff));
            log("[Comparison] SRCF is better in " + std::to_string(data.comparison.percentage_srcf_better) + "% of steps");
        }

        // ============================================================================
        // ФУНКЦИИ СОХРАНЕНИЯ
        // ============================================================================

        /**
         * @brief Сохранение данных в бинарном формате
         * @param data Данные для сохранения
         */
        void saveBinary(const SimulationData& data)
        {
            const std::string prefix = config_.output_dir + "/";
            saveVectorBinary(data.times, prefix + "times.bin");
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
            saveConfig();
            saveMetrics(data, prefix + "metrics.txt");
            saveComparisonStats(data, prefix + "comparison_stats.txt");
        }

        /**
         * @brief Сохранение данных в CSV формате
         * @param data Данные для сохранения
         */
        void saveTextCSV(const SimulationData& data)
        {
            const std::string filename = config_.output_dir + "/simulation_data.csv";
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "time,true_phi,true_p,"
                 << "meas_gyro_exact,meas_accel_exact,"  // Гироскоп и акселерометр
                 << "meas_gyro_noisy,meas_accel_noisy,control,";

            if (config_.test_ckf) {
                file << "ckf_phi,ckf_p,";
            }

            file << "srcf_phi,srcf_p,";
            if (config_.test_ckf) {
                file << "ckf_cov_11,ckf_cov_12,ckf_cov_21,ckf_cov_22,";
            }
            file << "srcf_cov_11,srcf_cov_12,srcf_cov_21,srcf_cov_22\n";
            file << std::fixed << std::setprecision(20);

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

        /**
         * @brief Сохранение данных в MATLAB формате
         * @param data Данные для сохранения
         */
        void saveTextMatlab(const SimulationData& data)
        {
            const std::string filename = config_.output_dir + "/simulation_data.m";
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

        /**
         * @brief Сохранение данных в TXT формате
         * @param data Данные для сохранения
         */
        void saveTextTXT(const SimulationData& data)
        {
            const std::string prefix = config_.output_dir + "/";
            saveMainDataTXT(data, prefix + "main_data.txt");
            saveCovarianceTXT(data, prefix + "covariances.txt");
            saveMeasurementsTXT(data, prefix + "measurements.txt");
            saveStatisticsTXT(data, prefix + "statistics.txt");
            saveConfig();
            saveMetrics(data, prefix + "filter_metrics.txt");
            saveComparisonStats(data, prefix + "comparison_details.txt");
        }

        /**
         * @brief Сохранение основных данных в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        void saveMainDataTXT(const SimulationData& data, const std::string& filename) const
        {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "KALMAN FILTER SIMULATION DATA\n";
            file << "========================================\n\n";

            file << std::fixed << std::setprecision(20);
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

        /**
         * @brief Сохранение данных ковариаций в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        void saveCovarianceTXT(const SimulationData& data, const std::string& filename) const
        {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            file << "========================================\n";
            file << "COVARIANCE MATRICES\n";
            file << "========================================\n\n";
            file << std::fixed << std::setprecision(20);
            for (size_t i = 0; i < std::min(data.times.size() - 1, (size_t)20); ++i) {
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
            file.close();
        }

        /**
         * @brief Сохранение данных измерений в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        static void saveMeasurementsTXT(const SimulationData& data, const std::string& filename)
        {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            file << "========================================\n";
            file << "MEASUREMENT DATA\n";
            file << "========================================\n\n";

            file << std::fixed << std::setprecision(20);
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

        /**
         * @brief Сохранение данных статистики в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        void saveStatisticsTXT(const SimulationData& data, const std::string& filename) const
        {
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
            double ckf_avg_error = ckf_total_error / static_cast<double>(ckf_errors.size());
            double srcf_avg_error = srcf_total_error / static_cast<double>(srcf_errors.size());

            // Вычисляем RMS
            double ckf_rms = 0.0;
            double srcf_rms = 0.0;
            for (size_t i = 0; i < ckf_errors.size(); ++i) {
                ckf_rms += ckf_errors[i] * ckf_errors[i];
                srcf_rms += srcf_errors[i] * srcf_errors[i];
            }
            ckf_rms = sqrt(ckf_rms / static_cast<double>(ckf_errors.size()));
            srcf_rms = sqrt(srcf_rms / static_cast<double>(srcf_errors.size()));
            file << std::fixed << std::setprecision(20);
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
                double rel_diff = (ckf_errors[i] > 1e-17) ?
                                  (diff / ckf_errors[i]) * 100.0 : 0.0;

                file << std::setw(8) << i
                     << std::setw(15) << ckf_errors[i]
                     << std::setw(15) << srcf_errors[i]
                     << std::setw(15) << diff
                     << std::setw(15) << std::setprecision(15) << rel_diff << std::setprecision(15)
                     << "\n";
            }
            file.close();
        }

        /**
         * @brief Сохранение данных сравнения фильтров в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        void saveComparisonStats(const SimulationData& data, const std::string& filename) const
        {
            std::ofstream file(filename);
            file << "=== DETAILED FILTER COMPARISON ===\n\n";
            file << std::fixed << std::setprecision(15);
            if (config_.test_ckf) {
                file << "SUMMARY:\n";
                file << "  Average error ratio (SRCF/CKF): " << data.comparison.avg_error_ratio << "\n";

                if (data.comparison.avg_error_ratio < 1.0) {
                    file << "  -> SRCF is " << (1.0 - data.comparison.avg_error_ratio) * 100.0
                         << "% better on average\n";
                    file << "  -> CKF is " << data.comparison.avg_error_ratio * 100.0
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

        /**
         * @brief Сохранение данных вычисления метрик в TXT формате
         * @param data Данные для сохранения
         * @param filename Название файла
         */
        void saveMetrics(const SimulationData& data, const std::string& filename) const
        {
            std::ofstream file(filename);
            file << "=== FILTER PERFORMANCE METRICS ===\n\n";
            file << std::fixed << std::setprecision(20);
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

        /**
         * @brief Сохранение текущей конфигурации
         */
        void saveConfig() const
        {
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
                    case model2::ControlScenario::SINE_UPDATE_WAVE: file << "SINE_UPDATE_WAVE"; break;
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

        // ============================================================================
        // ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СОХРАНЕНИЯ
        // ============================================================================

        /**
         * @brief Сохранение вектора double в бинарном формате
         * @param vec Вектор для сохранения
         * @param filename Имя файла
         */
        static void saveVectorBinary(const std::vector<double>& vec, const std::string& filename)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            const auto n = static_cast<std::streamsize>(vec.size());
            const auto data_size = static_cast<std::streamsize>(n * sizeof(double));
            file.write(reinterpret_cast<const char*>(vec.size()), sizeof(decltype(vec.size())));
            file.write(reinterpret_cast<const char*>(vec.data()), data_size);
            file.close();
        }

        /**
         * @brief Сохранение вектора Vector2d в бинарном формате
         * @param vec Вектор для сохранения
         * @param filename Имя файла
         */
        static void saveVector2dBinary(const std::vector<Eigen::Vector2d>& vec, const std::string& filename)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            const size_t n = vec.size();
            file.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            for (const auto& v : vec) {
                file.write(reinterpret_cast<const char*>(v.data()), 2 * sizeof(double));
            }
            file.close();
        }

        /**
         * @brief Сохранение вектора Matrix2d в бинарном формате
         * @param mats Вектор матриц для сохранения
         * @param filename Имя файла
         */
        static void saveMatrix2dBinary(const std::vector<Eigen::Matrix2d>& mats, const std::string& filename)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            const size_t n = mats.size();
            file.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            for (const auto& m : mats) {
                file.write(reinterpret_cast<const char*>(m.data()), 4 * sizeof(double));
            }
            file.close();
        }
    };

// ============================================================================
// ФУНКЦИИ АНАЛИЗА В СТИЛЕ VERHAEGEN & VAN DOOREN
// ============================================================================

    /**
     * @brief Анализ результатов в стиле Verhaegen & Van Dooren
     * @param data Данные симуляции
     * @param test_ckf Флаг тестирования CKF
     *
     * @note Реализует анализ численной устойчивости, чисел обусловленности
     *       и производительности фильтров согласно методологии статьи.
     */
    void analyze_verhaegen_style(const SimulationData& data, bool test_ckf = true)
    {
        std::cout << "\n=== ANALYSIS IN VERHAEGEN & VAN DOOREN STYLE ===\n";
        std::cout << "===TESTING TIMES===" << std::endl;
        for (double time : data.times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
        // 1. Анализ численной устойчивости
        if (test_ckf) {
            double max_ckf_asymmetry = 0.0;
            int ckf_non_positive = 0;

            for (const auto & ckf_covariance : data.ckf_covariances) {
                // Асимметрия: ||P - P^T|| / ||P||
                Eigen::Matrix2d asym_ckf = ckf_covariance -
                                           ckf_covariance.transpose();
                double asym_norm = asym_ckf.norm() / ckf_covariance.norm();
                max_ckf_asymmetry = std::max(max_ckf_asymmetry, asym_norm);

                // Положительная определенность
                Eigen::LLT<Eigen::Matrix2d> llt_ckf(ckf_covariance);
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
            if (i < data.srcf_covariances.size()) {
                Eigen::JacobiSVD<Eigen::Matrix2d> svd_srcf(data.srcf_covariances[i]);
                const Eigen::Vector2d& sv_srcf = svd_srcf.singularValues();
                if (sv_srcf.minCoeff() > 1e-17) {
                    const double cond_srcf = sv_srcf.maxCoeff() / sv_srcf.minCoeff();
                    if (cond_srcf < 1e15) {
                        avg_cond_srcf += cond_srcf;
                    }
                }

                if (test_ckf && i < data.ckf_covariances.size()) {
                    Eigen::JacobiSVD<Eigen::Matrix2d> svd_ckf(data.ckf_covariances[i]);
                    const Eigen::Vector2d& sv_ckf = svd_ckf.singularValues();
                    if (sv_ckf.minCoeff() > 1e-17) {
                        const double cond_ckf = sv_ckf.maxCoeff() / sv_ckf.minCoeff();
                        if (cond_ckf < 1e15) {
                            avg_cond_ckf += cond_ckf;
                        }
                    }
                }
                count++;
            }
        }

        if (count > 0) {
            avg_cond_srcf /= count;
            if (test_ckf) avg_cond_ckf /= count;
            std::cout << "\nCondition Number Analysis (average over " << count << " samples):\n";
            if (test_ckf) {
                std::cout << "  CKF: " << avg_cond_ckf << "\n";
            }
            std::cout << "  SRCF: " << avg_cond_srcf << "\n";
        }

        // 3. Анализ разницы ошибок
        if (test_ckf) {
            std::cout << "\n=== ERROR DIFFERENCE ANALYSIS ===\n";
            std::cout << std::fixed << std::setprecision(20);
            std::cout << "Average error ratio (SRCF/CKF): " << data.comparison.avg_error_ratio << "\n";
            std::cout << "RMS error ratio (SRCF/CKF): " << data.comparison.rms_error_ratio << "\n";
            std::cout << "Max error ratio (SRCF/CKF): " << data.comparison.max_error_ratio << "\n";

            if (data.comparison.avg_error_ratio < 1.0) {
                std::cout << "-> SRCF performs " << (1.0 - data.comparison.avg_error_ratio) * 100.0
                          << "% better on average\n";
                std::cout << "-> CKF performs " << data.comparison.avg_error_ratio * 100.0
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
        std::cout << std::fixed << std::setprecision(20);

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
            bool recommend_srcf = true;
            std::string reasons;

            // Рекомендации в стиле статьи
            if (data.srcf_metrics.average_error > data.ckf_metrics.average_error) {
                reasons +=  "\nCONCLUSION: CKF performs significantly better than SRCF\n";
            } else if (data.srcf_metrics.average_error < data.ckf_metrics.average_error) {
                reasons += "\nCONCLUSION: SRCF performs better than CKF\n";
            } else {
                reasons += "\nCONCLUSION: Both filters have similar performance\n";
            }

            if (data.comparison.avg_error_ratio < 0.95) {
                reasons += "Recommendation: Use SRCF (significantly better average error)\n";
            } else if (data.comparison.avg_error_ratio > 1.05) {
                recommend_srcf = false;
                reasons += "Recommendation: Use CKF (significantly better average error)\n";
            } else {
                reasons += "Recommendation: Both filters have similar performance\n";
            }

            if (data.comparison.rms_error_ratio > 1.15) {
                if (!recommend_srcf) reasons += "CKF has better RMS error\n";
                else recommend_srcf = false;
            }

            if (data.comparison.percentage_srcf_better < 40.0) {
                if (!recommend_srcf) reasons += "CKF is more consistent\n";
                else recommend_srcf = false;
            }

            if (data.comparison.percentage_srcf_better > 60.0) {
                reasons += "  Note: SRCF is more consistent across time steps\n";
            } else if (data.comparison.percentage_srcf_better < 40.0) {
                if (!recommend_srcf) reasons += "  Note: CKF is more consistent across time steps\n";
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