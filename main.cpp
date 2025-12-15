/**
 * @file main.cpp
 * @brief Тестовый набор для сравнения фильтров Калмана
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * @note Основной файл для тестирования и сравнения классического (CKF)
 *       и квадратно-корневого (SRCF) фильтров Калмана. Включает тесты
 *       в стиле Verhaegen & Van Dooren (1986) и различные сценарии.
 */

#include "data_generator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <filesystem>

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

/**
 * @brief Генерация данных во всех поддерживаемых форматах
 *
 * @note Создает данные в форматах BINARY, CSV, MATLAB и TXT
 *       для сравнения размеров файлов и удобства использования.
 */
void generate_all_formats()
{
    std::cout << "\n=== Generating Data in All Formats ===\n";
    data_generator::SimulationConfig base_config;
    base_config.total_steps = 1000;
    base_config.base_dt = 0.01;
    base_config.add_process_noise = true;
    base_config.add_measurement_noise = true;
    base_config.process_noise_scale = 1.0;
    base_config.measurement_noise_scale = 1.0;
    base_config.model_type = data_generator::ModelType::MODEL2;
    base_config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    base_config.time_mode = time_generator::TimeMode::UNIFORM;
    base_config.use_custom_initial = false;
    base_config.seed = 12345;

    // Создаем корневую директорию
    std::string root_dir = "./data/all_formats";

    // Кроссплатформенное создание директории
#ifdef _WIN32
    std::string command = "mkdir \"" + root_dir + "\"";
#else
    std::string command = "mkdir -p \"" + root_dir + "\"";
#endif
    system(command.c_str());

    // Массив всех форматов
    std::vector<std::pair<data_generator::DataFormat, std::string>> formats = {
            {data_generator::DataFormat::BINARY, "binary"},
            {data_generator::DataFormat::TEXT_CSV, "csv"},
            {data_generator::DataFormat::TEXT_MATLAB, "matlab"},
            {data_generator::DataFormat::TEXT_TXT, "txt"}
    };

    for (const auto& [format, name] : formats) {
        auto config = base_config;
        config.format = format;
        config.output_dir = root_dir;
        config.output_dir += "/";
        config.output_dir += name;

        std::cout << "\nGenerating " << name << " format...\n";

        // Кроссплатформенное создание поддиректории
#ifdef _WIN32
        command = "mkdir \"" + config.output_dir + "\" 2>nul";
#else
        command = "mkdir -p \"" + config.output_dir + "\"";
#endif
        system(command.c_str());

        try {
            data_generator::DataGenerator gen(config, base_config.seed);
            auto data = gen.generate();
            gen.save(data);

            std::cout << "  ✓ Saved to: " << config.output_dir << "\n";

            // Показать созданные файлы
            std::cout << "  Files created:\n";
            for (const auto& entry : std::filesystem::directory_iterator(config.output_dir)) {
                if (entry.is_regular_file()) {
                    auto file_size = entry.file_size();
                    std::cout << "    • " << entry.path().filename().string()
                              << " (" << file_size << " bytes)\n";
                }
            }

            // Для TXT формата можно проверить читаемость
            if (format == data_generator::DataFormat::TEXT_TXT) {
                std::string main_file = config.output_dir + "/main_data.txt";
                if (std::filesystem::exists(main_file)) {
                    std::ifstream test_file(main_file);
                    if (test_file.is_open()) {
                        std::string first_line;
                        std::getline(test_file, first_line);
                        std::cout << "  First line: " << first_line.substr(0, 50) << "...\n";
                        test_file.close();
                    }
                }
            }

            std::cout << "  Files created:\n";

            // Кроссплатформенный вывод содержимого директории
#ifdef _WIN32
            command = "dir \"" + config.output_dir + "\" /B";
#else
            command = "ls \"" + config.output_dir + "\"";
#endif

            std::cout << "  Executing: " << command << "\n";
            system(command.c_str());

        } catch (const std::exception& e) {
            std::cout << "  ✗ ERROR: " << e.what() << "\n";
        }
    }

    std::cout << "\n=== Comparison of Formats ===\n";
    std::cout << "Binary:   Fastest, smallest files, not human-readable\n";
    std::cout << "CSV:      Good for Excel/Spreadsheet analysis\n";
    std::cout << "MATLAB:   Ready for MATLAB/Octave import\n";
    std::cout << "TXT:      Human-readable, good for debugging\n";

    // Сравнение форматов
    std::cout << "\n=== FORMAT COMPARISON ===\n";

    // Сравним размеры файлов
    std::string binary_dir = "./data/all_formats/binary";
    std::string txt_dir = "./data/all_formats/txt";

    if (std::filesystem::exists(binary_dir) && std::filesystem::exists(txt_dir)) {
        size_t binary_total = 0;
        size_t txt_total = 0;

        for (const auto& entry : std::filesystem::directory_iterator(binary_dir)) {
            if (entry.is_regular_file()) {
                binary_total += entry.file_size();
            }
        }

        for (const auto& entry : std::filesystem::directory_iterator(txt_dir)) {
            if (entry.is_regular_file()) {
                txt_total += entry.file_size();
            }
        }

        std::cout << "Binary total size: " << binary_total << " bytes\n";
        std::cout << "TXT total size: " << txt_total << " bytes\n";
        if (binary_total > 0) {
            const double compression_ratio = static_cast<double>(txt_total) / static_cast<double>(binary_total);
            std::cout << "Compression ratio (TXT/Binary): " << compression_ratio << "\n";
        }
    }
}

/**
 * @brief Тестирование чтения/записи TXT формата
 */
void test_txt_format()
{
    std::cout << "\n=== TESTING TXT FORMAT READ/WRITE ===\n";

    data_generator::SimulationConfig config;
    config.total_steps = 500;
    config.base_dt = 0.02;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 1.0;
    config.measurement_noise_scale = 1.0;
    config.model_type = data_generator::ModelType::MODEL2;
    config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
    config.format = data_generator::DataFormat::TEXT_TXT;
    config.use_custom_initial = false;
    config.seed = 999;
    config.output_dir = "./data/txt_test";

    // Создаем директорию
    std::filesystem::create_directories(config.output_dir);

    try {
        // Генерация данных
        data_generator::DataGenerator gen(config, config.seed);
        auto sim_data = gen.generate();
        gen.save(sim_data);

        std::cout << "TXT data saved to: " << config.output_dir << "\n";

        // Проверка файлов
        std::cout << "\nChecking generated files:\n";
        for (const auto& entry : std::filesystem::directory_iterator(config.output_dir)) {
            if (entry.is_regular_file()) {
                std::cout << "  • " << entry.path().filename().string()
                          << " (" << entry.file_size() << " bytes)\n";
            }
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

// ============================================================================
// ОБЩАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
// ============================================================================

/**
 * @brief Запуск отдельного теста с сохранением результатов
 *
 * @param test_name Название теста
 * @param config Конфигурация симуляции
 * @param seed Зерно для генератора случайных чисел
 */
void run_test(const std::string& test_name,
              const data_generator::SimulationConfig& config,
              int seed = 42)
{
    std::cout << "\n=== " << test_name << " ===\n";

    // Создаем директорию
    std::filesystem::create_directories(config.output_dir);
    data_generator::SimulationConfig local_config = config;
    local_config.seed = seed;

    try {
        data_generator::DataGenerator gen(local_config, seed);
        auto data = gen.generate();
        gen.save(data);
        data_generator::analyze_verhaegen_style(data, config.test_ckf);
        if (config.test_ckf && !data.ckf_estimates.empty()) {
            const std::string csv_file = config.output_dir + "/comparison.csv";
            std::ofstream comparison_file(csv_file);
            comparison_file << "Step,Time,CKF_Error,SRCF_Error,CKF_Cov_Norm,SRCF_Cov_Norm,"
                            << "CKF_Cond_Number,SRCF_Cond_Number,Error_Diff,Rel_Diff_Percent,Innovation\n";

            // Используем существующие данные из comparison метрик
            for (size_t i = 0; i < data.true_states.size(); ++i) {
                // Время (используем i-ю временную метку)
                double time = (i < data.times.size()) ? data.times[i] : 0.0;

                // Ошибки (из error_history)
                double ckf_error = (i < data.ckf_metrics.error_history.size()) ?
                                   data.ckf_metrics.error_history[i] : 0.0;
                double srcf_error = (i < data.srcf_metrics.error_history.size()) ?
                                    data.srcf_metrics.error_history[i] : 0.0;

                // Нормы ковариаций
                double ckf_cov_norm = (i < data.ckf_covariances.size()) ?
                                      data.ckf_covariances[i].norm() : 0.0;
                double srcf_cov_norm = (i < data.srcf_covariances.size()) ?
                                       data.srcf_covariances[i].norm() : 0.0;

                // Числа обусловленности
                double ckf_cond = 1.0;
                double srcf_cond = 1.0;
                if (i < data.ckf_covariances.size()) {
                    Eigen::JacobiSVD<Eigen::Matrix2d> svd_ckf(data.ckf_covariances[i]);
                    auto sv_ckf = svd_ckf.singularValues();
                    if (sv_ckf.minCoeff() > 1e-15 && sv_ckf.maxCoeff() > 1e-15) {
                        ckf_cond = sv_ckf.maxCoeff() / sv_ckf.minCoeff();
                    }
                }
                if (i < data.srcf_covariances.size()) {
                    Eigen::JacobiSVD<Eigen::Matrix2d> svd_srcf(data.srcf_covariances[i]);
                    auto sv_srcf = svd_srcf.singularValues();
                    if (sv_srcf.minCoeff() > 1e-15 && sv_srcf.maxCoeff() > 1e-15) {
                        srcf_cond = sv_srcf.maxCoeff() / sv_srcf.minCoeff();
                    }
                }

                // Разница ошибок
                double error_diff = srcf_error - ckf_error;
                double rel_diff = (ckf_error > 1e-15) ?
                                  (error_diff / ckf_error) * 100.0 : 0.0;

                double innovation = 0.0;

                if (i > 0) {
                    const Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
                    innovation = innov.norm();
                }

                comparison_file << i << "," << std::setprecision(20) << time << ","
                                << ckf_error << "," << srcf_error << ","
                                << ckf_cov_norm << "," << srcf_cov_norm << ","
                                << ckf_cond << "," << srcf_cond << ","
                                << error_diff << "," << rel_diff << "," << innovation << "\n";
            }
            comparison_file.close();
        }
        std::cout << "Results saved to: " << config.output_dir << "\n";
    } catch (const std::exception& e) {
        std::cout << "ERROR in test '" << test_name << "': " << e.what() << "\n";
    }
}

// ============================================================================
// ТЕСТ 1: РАЗНЫЕ УРОВНИ ШУМА (как в статье Verhaegen & Van Dooren)
// ============================================================================

/**
 * @brief Тесты с разными уровнями шума (статья Verhaegen & Van Dooren)
 */
void run_verhaegen_tests()
{
    std::cout << "\n=== TEST SERIES 1: NOISE LEVEL TESTS (Verhaegen Style) ===\n";
    std::vector<std::pair<double, double>> noise_levels = {
            {0.1, 0.1},     // Test 1: Low noise
            {1.0, 1.0},     // Test 2: Moderate noise
            {10.0, 10.0},   // Test 3: High noise
            {0.01, 1.0},    // Test 4: Low process, high measurement noise
            {1.0, 0.01},    // Test 5: High process, low measurement noise
    };

    // Сводный файл
    std::ofstream summary_file("verhaegen_summary.csv");
    summary_file << "Test,ProcessNoise,MeasNoise,CKF_AvgError,SRCF_AvgError,CKF_RMS,SRCF_RMS,"
                 << "CKF_Asymmetry,SRCF_Asymmetry,CKF_PosDef,SRCF_PosDef\n";

    for (size_t test_num = 0; test_num < noise_levels.size(); ++test_num) {
        auto [process_scale, meas_scale] = noise_levels[test_num];
        data_generator::SimulationConfig config;
        config.total_steps = 2000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.process_noise_scale = process_scale;
        config.measurement_noise_scale = meas_scale;
        config.model_type = data_generator::ModelType::MODEL2;
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::BINARY;
        config.use_custom_initial = false;
        config.seed = 12345 + static_cast<int>(test_num);
        config.output_dir = "./data/verhaegen_test_" + std::to_string(test_num + 1);
        std::string test_name = "Verhaegen Test " + std::to_string(test_num + 1) +
                                " (Proc=" + std::to_string(process_scale) +
                                ", Meas=" + std::to_string(meas_scale) + ")";
        run_test(test_name, config, config.seed);
        try {
            // Для сводного файла нужно прочитать метрики
            data_generator::DataGenerator gen(config, 12345 + static_cast<int>(test_num));
            auto data = gen.generate();

            // Рассчитываем метрики для сводного файла
            double ckf_avg = 0.0, srcf_avg = 0.0;
            double ckf_rms = 0.0, srcf_rms = 0.0;
            double ckf_asym = 0.0, srcf_asym = 0.0;
            int ckf_posdef = 0, srcf_posdef = 0;

            for (size_t i = 0; i < data.true_states.size(); ++i) {
                const double ckf_err = (data.true_states[i] - data.ckf_estimates[i]).norm();
                const double srcf_err = (data.true_states[i] - data.srcf_estimates[i]).norm();

                ckf_avg += ckf_err;
                srcf_avg += srcf_err;
                ckf_rms += ckf_err * ckf_err;
                srcf_rms += srcf_err * srcf_err;

                const Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] -
                                                 data.ckf_covariances[i].transpose();
                const Eigen::Matrix2d asym_srcf = data.srcf_covariances[i] -
                                                  data.srcf_covariances[i].transpose();

                ckf_asym = std::max(ckf_asym, asym_ckf.norm() / data.ckf_covariances[i].norm());
                srcf_asym = std::max(srcf_asym, asym_srcf.norm() / data.srcf_covariances[i].norm());

                Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
                Eigen::LLT<Eigen::Matrix2d> llt_srcf(data.srcf_covariances[i]);

                if (llt_ckf.info() == Eigen::Success) ckf_posdef++;
                if (llt_srcf.info() == Eigen::Success) srcf_posdef++;
            }

            ckf_avg /= static_cast<double>(data.true_states.size());
            srcf_avg /= static_cast<double>(data.true_states.size());
            ckf_rms = std::sqrt(ckf_rms / static_cast<double>(data.true_states.size()));
            srcf_rms = std::sqrt(srcf_rms / static_cast<double>(data.true_states.size()));

            summary_file << test_num + 1 << ","
                         << process_scale << "," << meas_scale << ","
                         << ckf_avg << "," << srcf_avg << ","
                         << ckf_rms << "," << srcf_rms << ","
                         << ckf_asym << "," << srcf_asym << ","
                         << ckf_posdef << "," << srcf_posdef << "\n";

        } catch (const std::exception& e) {
            std::cout << "WARNING: Could not generate summary for test "
                      << test_num + 1 << ": " << e.what() << "\n";
        }
    }

    summary_file.close();
    std::cout << "\nSummary saved to: verhaegen_summary.csv\n";
}

// ============================================================================
// ТЕСТ 2: РАЗНЫЕ ФОРМАТЫ ДАННЫХ
// ============================================================================

/**
 * @brief Тестирование всех поддерживаемых форматов данных
 */
void test_different_formats()
{
    std::cout << "\n=== TEST SERIES 2: DIFFERENT DATA FORMATS ===\n";
    data_generator::SimulationConfig base_config;
    base_config.total_steps = 1000;
    base_config.base_dt = 0.01;
    base_config.add_process_noise = true;
    base_config.add_measurement_noise = true;
    base_config.model_type = data_generator::ModelType::MODEL2;
    base_config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    base_config.time_mode = time_generator::TimeMode::UNIFORM;
    base_config.use_custom_initial = false;
    base_config.seed = 54321;

    // Тестируем все форматы
    std::vector<std::pair<data_generator::DataFormat, std::string>> formats = {
            {data_generator::DataFormat::BINARY, "binary"},
            {data_generator::DataFormat::TEXT_CSV, "csv"},
            {data_generator::DataFormat::TEXT_MATLAB, "matlab"},
            {data_generator::DataFormat::TEXT_TXT, "txt"}
    };

    for (const auto& [format, name] : formats) {
        auto config = base_config;
        config.format = format;
        config.output_dir = "./data/format_test_" + name;
        std::string test_name = "Format Test: " + name;
        run_test(test_name, config, base_config.seed);
    }
}

// ============================================================================
// ТЕСТ 3: РАЗНЫЕ СЦЕНАРИИ УПРАВЛЕНИЯ
// ============================================================================
/**
 * @brief Тестирование различных сценариев управления
 */
void test_different_scenarios()
{
    std::cout << "\n=== TEST SERIES 3: DIFFERENT CONTROL SCENARIOS ===\n";

    std::vector<std::pair<model2::ControlScenario, std::string>> scenarios = {
            {model2::ControlScenario::ZERO_HOLD, "Zero Hold"},
            {model2::ControlScenario::STEP_MANEUVER, "Step Maneuver"},
            {model2::ControlScenario::SINE_WAVE, "Sine Wave"},
            {model2::ControlScenario::PULSE, "Pulse"}
    };

    for (const auto& [scenario, name] : scenarios) {
        data_generator::SimulationConfig config;
        config.total_steps = 2000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.model_type = data_generator::ModelType::MODEL2;
        config.scenario.scenario2 = scenario;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::BINARY;
        config.use_custom_initial = false;
        config.output_dir = "./data/scenario_" + name;
        config.seed = 67890 + static_cast<int>(scenario);
        std::string test_name = "Scenario: " + name;
        run_test(test_name, config, config.seed);
    }
}

// ============================================================================
// ТЕСТ 4: РАЗНЫЕ РЕЖИМЫ ВРЕМЕНИ
// ============================================================================

/**
 * @brief Тестирование различных режимов генерации временных меток
 */
void test_different_time_modes()
{
    std::cout << "\n=== TEST SERIES 4: DIFFERENT TIME MODES ===\n";
    std::vector<std::pair<time_generator::TimeMode, std::string>> time_modes = {
            {time_generator::TimeMode::UNIFORM, "Uniform"},
            {time_generator::TimeMode::VARIABLE, "Variable"},
            {time_generator::TimeMode::RANDOM_JITTER, "Random Jitter"}
    };

    for (const auto& [mode, name] : time_modes) {
        data_generator::SimulationConfig config;
        config.total_steps = 2000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.model_type = data_generator::ModelType::MODEL2;
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.time_mode = mode;
        config.format = data_generator::DataFormat::BINARY;
        config.use_custom_initial = false;
        config.output_dir = "./data/time_mode_" + name;
        config.seed = 33333 + static_cast<int>(mode);
        std::string test_name = "Time Mode: " + name;
        run_test(test_name, config, config.seed);
    }
}

// ============================================================================
// ТЕСТ 5: НЕУСТОЙЧИВАЯ СИСТЕМА (как в статье)
// ============================================================================

/**
 * @brief Тестирование неустойчивой системы (увеличенный шум процесса)
 */
void test_unstable_system()
{
    std::cout << "\n=== TEST 5: UNSTABLE SYSTEM (Verhaegen Style) ===\n";
    data_generator::SimulationConfig config;
    config.total_steps = 2000;
    config.base_dt = 0.01;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 5.0;      // Увеличенный шум процесса
    config.measurement_noise_scale = 0.1;  // Низкий шум измерений
    config.model_type = data_generator::ModelType::MODEL2;
    config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    config.time_mode = time_generator::TimeMode::UNIFORM;
    config.format = data_generator::DataFormat::TEXT_CSV;  // Для анализа
    config.use_custom_initial = false;
    config.output_dir = "./data/unstable_test";
    config.seed = 99999;
    run_test("Unstable System Test", config, config.seed);
}

// ============================================================================
// ТЕСТ 6: СРАВНЕНИЕ МОДЕЛЕЙ
// ============================================================================

/**
 * @brief Сравнение производительности фильтров на разных моделях
 */
void test_model_comparison()
{
    std::cout << "\n=== TEST 6: MODEL COMPARISON ===\n";
    std::cout << "NOTE: Model 0 support needs to be added to DataGenerator.\n";
    std::cout << "Currently only Model 2 is supported.\n";

    // Тест с Model 0 (упрощенная модель рыскания)
    {
        std::cout << "\n--- Model 0 Test (Simplified Yaw Model) ---\n";
        data_generator::SimulationConfig config;
        config.total_steps = 1000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.model_type = data_generator::ModelType::MODEL0;
        config.scenario.scenario0 = model0::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::TEXT_CSV;
        config.use_custom_initial = false;
        config.output_dir = "./data/model0_test";
        config.test_ckf = true;
        config.seed = 11111;
        run_test("Model 0 Test (Yaw Model)", config, config.seed);
    }

    // Тест с Model 2 (модель крена)
    {
        std::cout << "\n--- Model 2 Test (Roll Model) ---\n";
        data_generator::SimulationConfig config;
        config.total_steps = 1000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.model_type = data_generator::ModelType::MODEL2;
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::TEXT_CSV;
        config.use_custom_initial = false;
        config.output_dir = "./data/model2_test";
        config.test_ckf = true;
        config.seed = 22222;
        run_test("Model 2 Test (Roll Model)", config, config.seed);
    }
}

// ============================================================================
// ТЕСТ 7: БАЗОВЫЙ ТЕСТ ДЛЯ БЫСТРОЙ ПРОВЕРКИ
// ============================================================================

/**
 * @brief Базовый тест для проверки работоспособности системы
 */
void test_basic()
{
    std::cout << "\n=== TEST 8: WITH PROCESS AND MEASUREMENT NOISE ===\n";
    data_generator::SimulationConfig config;
    config.total_steps = 100;  // Увеличим для статистики
    config.base_dt = 0.02;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 1.0;      // Стандартный масштаб
    config.measurement_noise_scale = 1.0;  // Стандартный масштаб
    config.model_type = data_generator::ModelType::MODEL2;
    config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
    config.time_mode = time_generator::TimeMode::UNIFORM;
    config.format = data_generator::DataFormat::TEXT_TXT;
    config.output_dir = "./data/test_with_noise";
    config.test_ckf = true;
    config.use_custom_initial = true;
    config.initial_state << 1, 2;
    config.initial_covariance << 0.01, 0.0,
            0.0, 0.01;
    config.seed = 77777;
    run_test("Test with Noise", config, config.seed);

//    std::cout << "\n=== TEST 11: PROCESS NOISE ONLY ===\n";
//
//    data_generator::SimulationConfig config;
//    config.total_steps = 150;
//    config.base_dt = 0.02;
//    config.add_process_noise = true;       // Только шум процесса
//    config.add_measurement_noise = false;  // Нет шума измерений
//    config.process_noise_scale = 3.0;      // Средний шум процесса
//    config.measurement_noise_scale = 0.0;
//    config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
//    config.time_mode = time_generator::TimeMode::UNIFORM;
//    config.format = data_generator::DataFormat::TEXT_CSV;
//    config.output_dir = "./data/test_process_noise";
//    config.use_custom_initial = false;
//    run_test("Process Noise Only", config, 11111);
}

// ============================================================================
// ТЕСТ 8: ПРОИЗВОДИТЕЛЬНОСТЬ (дополнительный тест)
// ============================================================================

/**
 * @brief Тест производительности с измерением времени выполнения
 */
void test_performance() {
    std::cout << "\n=== TEST 8: PERFORMANCE TEST ===\n";

    const std::chrono::time_point<std::chrono::high_resolution_clock> start_time =
            std::chrono::high_resolution_clock::now();

    data_generator::SimulationConfig config;
    config.total_steps = 5000;  // Большое количество шагов для теста производительности
    config.base_dt = 0.01;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.model_type = data_generator::ModelType::MODEL2;
    config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    config.time_mode = time_generator::TimeMode::UNIFORM;
    config.format = data_generator::DataFormat::TEXT_CSV;  // Самый быстрый формат
    config.use_custom_initial = false;
    config.output_dir = "./data/performance_test";
    config.test_ckf = true;
    config.seed = 88888;

    try {
        run_test("Performance test", config, config.seed);

        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Performance results:\n";
        std::cout << "  Steps processed: " << config.total_steps << "\n";
        std::cout << "  Total time: " << duration.count() << " ms\n";
        std::cout << "  Time per step: " << static_cast<double>(duration.count()) / static_cast<double>(config.total_steps) << " ms\n";
        std::cout << "  Data saved to: " << config.output_dir << "\n";

    } catch (const std::exception& e) {
        std::cout << "Performance test failed: " << e.what() << "\n";
    }
}

// ============================================================================
// ОСНОВНАЯ ФУНКЦИЯ С МЕНЮ ВЫБОРА
// ============================================================================

/**
 * @brief Отображение меню выбора тестов
 */
void display_menu()
{
    std::cout << "\nSelect test series to run:\n";
    std::cout << "1. All tests (comprehensive)\n";
    std::cout << "2. Verhaegen noise tests only\n";
    std::cout << "3. Format comparison tests\n";
    std::cout << "4. Control scenario tests\n";
    std::cout << "5. Time mode tests\n";
    std::cout << "6. Model comparison tests\n";
    std::cout << "7. Basic functionality test\n";
    std::cout << "8. Performance test\n";
    std::cout << "9. Unstable system\n";
    std::cout << "10. Custom test\n";
    std::cout << "0. Exit\n";
    std::cout << "Enter choice (0-9): ";
}

/**
 * @brief Обработка пользовательского ввода для выбора теста
 * @return int Выбор пользователя
 */
int get_user_choice()
{
    int choice = -1;
    std::string input;
    while (true) {
        std::getline(std::cin, input);
        std::stringstream ss(input);
        if (ss >> choice && choice >= 0 && choice <= 10) {
            return choice;
        }
        std::cout << "Invalid input. Please enter a number between 0 and 10: ";
    }
}

/**
 * @brief Создание пользовательской конфигурации теста
 * @return data_generator::SimulationConfig Пользовательская конфигурация
 */
data_generator::SimulationConfig create_custom_config()
{
    data_generator::SimulationConfig config;
    std::cout << "\n=== CUSTOM TEST CONFIGURATION ===\n";

    // Количество шагов
    std::cout << "Enter total steps (100-10000): ";
    std::cin >> config.total_steps;
    config.total_steps = std::max(100, std::min(10000, static_cast<int>(config.total_steps)));

    // Шаг времени
    std::cout << "Enter base dt (0.001-0.1): ";
    std::cin >> config.base_dt;
    config.base_dt = std::max(0.001, std::min(0.1, config.base_dt));

    // Модель
    std::cout << "Select model (0=MODEL0, 1=MODEL2): ";
    int model_choice;
    std::cin >> model_choice;
    config.model_type = (model_choice == 0) ?
                        data_generator::ModelType::MODEL0 : data_generator::ModelType::MODEL2;

    // Сценарий
    std::cout << "Select scenario (0=ZERO_HOLD, 1=STEP, 2=SINE, 3=PULSE): ";
    int scenario_choice;
    std::cin >> scenario_choice;

    if (config.model_type == data_generator::ModelType::MODEL0) {
        config.scenario.scenario0 = static_cast<model0::ControlScenario>(scenario_choice);
    } else {
        config.scenario.scenario2 = static_cast<model2::ControlScenario>(scenario_choice);
    }

    // Формат
    std::cout << "Select format (0=BINARY, 1=CSV, 2=MATLAB, 3=TXT): ";
    int format_choice;
    std::cin >> format_choice;
    config.format = static_cast<data_generator::DataFormat>(format_choice);

    // Тестирование CKF
    std::cout << "Test CKF filter? (0=No, 1=Yes): ";
    int test_ckf_choice;
    std::cin >> test_ckf_choice;
    config.test_ckf = (test_ckf_choice != 0);
    config.output_dir = "./data/custom_test";
    std::cout << "Choose random seed for testing: ";
    int seeed;
    std::cin >> seeed;
    config.seed = seeed;

    // Очистка буфера ввода
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return config;
}

// ============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "   KALMAN FILTER COMPARISON SUITE\n";
    std::cout << "   Based on Verhaegen & Van Dooren (1986)\n";
    std::cout << "   Version 2.0\n";
    std::cout << "========================================\n\n";

    // Создаем корневую директорию для данных
    std::filesystem::create_directories("./data");
//    generate_all_formats();
//    test_txt_format();

    // Выбор тестов для запуска
    int choice;
    bool exit_program = false;

    while (!exit_program) {
        display_menu();
        choice = get_user_choice();

        switch (choice) {
            case 0:  // Выход
                exit_program = true;
                break;

            case 1:  // Все тесты
                std::cout << "\nRunning all test series...\n";
                test_basic();
                run_verhaegen_tests();
                test_different_formats();
                test_different_scenarios();
                test_different_time_modes();
                test_unstable_system();
                test_model_comparison();
                test_performance();
                break;

            case 2:  // Тесты Verhaegen
                std::cout << "\nRunning Verhaegen noise tests...\n";
                run_verhaegen_tests();
                break;

            case 3:  // Тесты форматов
                std::cout << "\nRunning format comparison tests...\n";
                test_different_formats();
                break;

            case 4:  // Тесты сценариев
                std::cout << "\nRunning control scenario tests...\n";
                test_different_scenarios();
                break;

            case 5:  // Тесты режимов времени
                std::cout << "\nRunning time mode tests...\n";
                test_different_time_modes();
                break;

            case 6:  // Сравнение моделей
                std::cout << "\nRunning model comparison tests...\n";
                test_model_comparison();
                break;

            case 7:  // Базовый тест
                std::cout << "\nRunning basic functionality test...\n";
                test_basic();
                break;

            case 8:  // Тест производительности
                std::cout << "\nRunning performance test...\n";
                test_performance();
                break;

            case 9:  // Тест нестабильной системы
                std::cout << "\nTesting unstable system...\n";
                test_unstable_system();
                break;

            case 10:  // Пользовательский тест
            {
                std::cout << "\nRunning custom test...\n";
                data_generator::SimulationConfig config = create_custom_config();
                run_test("Custom Test", config, 88888);
            }
                break;
            default:
                std::cout << "Invalid choice. Running basic test...\n";
                test_basic();
                break;
        }

        if (!exit_program) {
            std::cout << "\nPress Enter to continue...";
            std::cin.get();
        }
    }

    // Отчет о завершении
    std::cout << "\n========================================\n";
    std::cout << "   TEST SUITE COMPLETED\n";
    std::cout << "========================================\n";
    std::cout << "Generated data saved in ./data/\n";
    std::cout << "Files include:\n";
    std::cout << "  • Simulation data (various formats)\n";
    std::cout << "  • Configuration files\n";
    std::cout << "  • Performance metrics\n";
    std::cout << "  • Comparison CSV files\n";
    std::cout << "  • Verhaegen-style analysis results\n";
    std::cout << "\nThank you for using Kalman Filter Comparison Suite!\n";
    return 0;
}