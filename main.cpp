
#include "data_generator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <filesystem>

void generate_all_formats() {
    std::cout << "\n=== Generating Data in All Formats ===\n";
    data_generator::SimulationConfig base_config;
    base_config.total_steps = 1000;
    base_config.base_dt = 0.01;
    base_config.add_process_noise = true;
    base_config.add_measurement_noise = true;
    base_config.process_noise_scale = 1.0;
    base_config.measurement_noise_scale = 1.0;
    base_config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    base_config.time_mode = time_generator::TimeMode::UNIFORM;

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
        config.output_dir = root_dir + "/" + name;

        std::cout << "\nGenerating " << name << " format...\n";

        // Кроссплатформенное создание поддиректории
#ifdef _WIN32
        command = "mkdir \"" + config.output_dir + "\" 2>nul";
#else
        command = "mkdir -p \"" + config.output_dir + "\"";
#endif
        system(command.c_str());

        try {
            data_generator::DataGenerator gen(config, 12345);
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
        std::cout << "Compression ratio (TXT/Binary): "
                  << (double)txt_total / binary_total << "\n";
    }

}

// Тестирование чтения TXT формата
void test_txt_format() {
    std::cout << "\n=== TESTING TXT FORMAT READ/WRITE ===\n";

    data_generator::SimulationConfig config;
    config.total_steps = 500;
    config.base_dt = 0.02;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 1.0;
    config.measurement_noise_scale = 1.0;
    config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
    config.format = data_generator::DataFormat::TEXT_TXT;
    config.output_dir = "./data/txt_test";

    // Создаем директорию
    std::filesystem::create_directories(config.output_dir);

    // Генерация данных
    data_generator::DataGenerator gen(config, 999);
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
}

// ============================================================================
// УНИВИЦИРОВАННАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА ТЕСТОВ
// ============================================================================
void run_test(const std::string& test_name,
              const data_generator::SimulationConfig& config,
              int seed = 42) {

    std::cout << "\n=== " << test_name << " ===\n";

    // Создаем директорию
    std::filesystem::create_directories(config.output_dir);

    // Генерируем данные
    data_generator::DataGenerator gen(config, seed);
    auto data = gen.generate();

    // Сохраняем данные
    gen.save(data);

    // Анализируем в стиле Verhaegen & Van Dooren
    data_generator::analyze_verhaegen_style(data);

    // Сохраняем сравнение в CSV
    std::string csv_file = config.output_dir + "/comparison.csv";

    std::ofstream comparison_file(csv_file);
    comparison_file << "Step,CKF_Error,SRCF_Error,CKF_Cov_Norm,SRCF_Cov_Norm,Innovation\n";

    for (size_t i = 0; i < data.true_states.size(); i += 10) {  // Сохраняем каждый 10-й шаг
        double ckf_error = (data.true_states[i] - data.ckf_estimates[i]).norm();
        double srcf_error = (data.true_states[i] - data.srcf_estimates[i]).norm();
        double ckf_cov_norm = data.ckf_covariances[i].norm();
        double srcf_cov_norm = data.srcf_covariances[i].norm();
        double innovation = 0.0;

        if (i > 0) {
            Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
            innovation = innov.norm();
        }

        comparison_file << i << ","
                        << ckf_error << ","
                        << srcf_error << ","
                        << ckf_cov_norm << ","
                        << srcf_cov_norm << ","
                        << innovation << "\n";
    }
    comparison_file.close();

    std::cout << "Results saved to: " << config.output_dir << "\n";
}

// ============================================================================
// ТЕСТ 1: РАЗНЫЕ УРОВНИ ШУМА (как в статье Verhaegen & Van Dooren)
// ============================================================================
void run_verhaegen_tests() {
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

    for (int test_num = 0; test_num < noise_levels.size(); ++test_num) {
        auto [process_scale, meas_scale] = noise_levels[test_num];

        data_generator::SimulationConfig config;
        config.total_steps = 2000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.process_noise_scale = process_scale;
        config.measurement_noise_scale = meas_scale;
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::BINARY;
        config.output_dir = "./data/verhaegen_test_" + std::to_string(test_num + 1);

        std::string test_name = "Verhaegen Test " + std::to_string(test_num + 1) +
                                " (Proc=" + std::to_string(process_scale) +
                                ", Meas=" + std::to_string(meas_scale) + ")";

        run_test(test_name, config, 12345 + test_num);

        // Для сводного файла нужно прочитать метрики
        data_generator::DataGenerator gen(config, 12345 + test_num);
        auto data = gen.generate();

        // Рассчитываем метрики для сводного файла
        double ckf_avg = 0.0, srcf_avg = 0.0;
        double ckf_rms = 0.0, srcf_rms = 0.0;
        double ckf_asym = 0.0, srcf_asym = 0.0;
        int ckf_posdef = 0, srcf_posdef = 0;

        for (size_t i = 0; i < data.true_states.size(); ++i) {
            double ckf_err = (data.true_states[i] - data.ckf_estimates[i]).norm();
            double srcf_err = (data.true_states[i] - data.srcf_estimates[i]).norm();

            ckf_avg += ckf_err;
            srcf_avg += srcf_err;
            ckf_rms += ckf_err * ckf_err;
            srcf_rms += srcf_err * srcf_err;

            Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] - data.ckf_covariances[i].transpose();
            Eigen::Matrix2d asym_srcf = data.srcf_covariances[i] - data.srcf_covariances[i].transpose();

            ckf_asym = std::max(ckf_asym, asym_ckf.norm() / data.ckf_covariances[i].norm());
            srcf_asym = std::max(srcf_asym, asym_srcf.norm() / data.srcf_covariances[i].norm());

            Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
            Eigen::LLT<Eigen::Matrix2d> llt_srcf(data.srcf_covariances[i]);

            if (llt_ckf.info() == Eigen::Success) ckf_posdef++;
            if (llt_srcf.info() == Eigen::Success) srcf_posdef++;
        }

        ckf_avg /= data.true_states.size();
        srcf_avg /= data.true_states.size();
        ckf_rms = sqrt(ckf_rms / data.true_states.size());
        srcf_rms = sqrt(srcf_rms / data.true_states.size());

        summary_file << test_num + 1 << ","
                     << process_scale << "," << meas_scale << ","
                     << ckf_avg << "," << srcf_avg << ","
                     << ckf_rms << "," << srcf_rms << ","
                     << ckf_asym << "," << srcf_asym << ","
                     << ckf_posdef << "," << srcf_posdef << "\n";
    }

    summary_file.close();
    std::cout << "\nSummary saved to: verhaegen_summary.csv\n";
}

// ============================================================================
// ТЕСТ 2: РАЗНЫЕ ФОРМАТЫ ДАННЫХ
// ============================================================================
void test_different_formats() {
    std::cout << "\n=== TEST SERIES 2: DIFFERENT DATA FORMATS ===\n";

    data_generator::SimulationConfig base_config;
    base_config.total_steps = 1000;
    base_config.base_dt = 0.01;
    base_config.add_process_noise = true;
    base_config.add_measurement_noise = true;
    base_config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    base_config.time_mode = time_generator::TimeMode::UNIFORM;

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
        run_test(test_name, config, 54321);
    }
}

// ============================================================================
// ТЕСТ 3: РАЗНЫЕ СЦЕНАРИИ УПРАВЛЕНИЯ
// ============================================================================
void test_different_scenarios() {
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
        config.scenario.scenario2 = scenario;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::BINARY;
        config.output_dir = "./data/scenario_" + name;

        std::string test_name = "Scenario: " + name;
        run_test(test_name, config, 67890 + static_cast<int>(scenario));
    }
}

// ============================================================================
// ТЕСТ 4: РАЗНЫЕ РЕЖИМЫ ВРЕМЕНИ
// ============================================================================
void test_different_time_modes() {
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
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.time_mode = mode;
        config.format = data_generator::DataFormat::BINARY;
        config.output_dir = "./data/time_mode_" + name;

        std::string test_name = "Time Mode: " + name;
        run_test(test_name, config, 33333 + static_cast<int>(mode));
    }
}

// ============================================================================
// ТЕСТ 5: НЕУСТОЙЧИВАЯ СИСТЕМА (как в статье)
// ============================================================================
void test_unstable_system() {
    std::cout << "\n=== TEST 5: UNSTABLE SYSTEM (Verhaegen Style) ===\n";

    data_generator::SimulationConfig config;
    config.total_steps = 2000;
    config.base_dt = 0.01;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 5.0;      // Увеличенный шум процесса
    config.measurement_noise_scale = 0.1;  // Низкий шум измерений
    config.scenario.scenario2 = model2::ControlScenario::STEP_MANEUVER;
    config.time_mode = time_generator::TimeMode::UNIFORM;
    config.format = data_generator::DataFormat::TEXT_CSV;  // Для анализа
    config.output_dir = "./data/unstable_test";

    run_test("Unstable System Test", config, 99999);
}

// ============================================================================
// ТЕСТ 6: МОДЕЛЬ 0 VS МОДЕЛЬ 2
// ============================================================================
void test_model_comparison() {
    std::cout << "\n=== TEST 6: MODEL COMPARISON ===\n";


    std::cout << "NOTE: Model 0 support needs to be added to DataGenerator.\n";
    std::cout << "Currently only Model 2 is supported.\n";

    // Тест с Model 0
    {
        data_generator::SimulationConfig config;
        config.model_type = data_generator::ModelType::MODEL0;
        config.scenario.scenario0 = model0::ControlScenario::SINE_WAVE;
        config.output_dir = "./data/model0_test";
        config.test_ckf = false;
        run_test("Model 0 Test", config, 11111);
    }

    // Тест с Model 2
    {
        data_generator::SimulationConfig config;
        config.model_type = data_generator::ModelType::MODEL2;
        config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
        config.output_dir = "./data/model2_test";
        config.test_ckf = false;
        run_test("Model 2 Test", config, 22222);
    }
}

// ============================================================================
// ТЕСТ 7: БАЗОВЫЙ ТЕСТ ДЛЯ БЫСТРОЙ ПРОВЕРКИ
// ============================================================================
void test_basic() {
    std::cout << "\n=== TEST 8: WITH PROCESS AND MEASUREMENT NOISE ===\n";

    data_generator::SimulationConfig config;
    config.total_steps = 100;  // Увеличим для статистики
    config.base_dt = 0.02;
    config.add_process_noise = true;
    config.add_measurement_noise = true;
    config.process_noise_scale = 1.0;      // Стандартный масштаб
    config.measurement_noise_scale = 1.0;  // Стандартный масштаб
    config.scenario.scenario2 = model2::ControlScenario::SINE_WAVE;
    config.time_mode = time_generator::TimeMode::UNIFORM;
    config.format = data_generator::DataFormat::TEXT_TXT;
    config.output_dir = "./data/test_with_noise";
    run_test("Test with Noise", config, 77777);
}

// ============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "   KALMAN FILTER TEST SUITE\n";
    std::cout << "   Based on Verhaegen & Van Dooren (1986)\n";
    std::cout << "========================================\n\n";

    // Создаем корневую директорию для данных
    std::filesystem::create_directories("./data");
    generate_all_formats();
    test_txt_format();

    // Выбор тестов для запуска
    int choice;
    std::cout << "Select test series to run:\n";
    std::cout << "1. All tests (comprehensive)\n";
    std::cout << "2. Verhaegen noise tests only\n";
    std::cout << "3. Format comparison tests\n";
    std::cout << "4. Basic functionality test\n";
    std::cout << "5. Test model comparison\n";
    std::cout << "6. Custom test\n";
    std::cout << "Enter choice (1-6): ";
    std::cin >> choice;

    switch (choice) {
        case 1:  // Все тесты
            std::cout << "\nRunning all test series...\n";
            test_basic();
            run_verhaegen_tests();
            test_different_formats();
            test_different_scenarios();
            test_different_time_modes();
            test_unstable_system();
            test_model_comparison();
            break;

        case 2:  // Только тесты Verhaegen
            std::cout << "\nRunning Verhaegen noise tests...\n";
            run_verhaegen_tests();
            break;

        case 3:  // Тесты форматов
            std::cout << "\nRunning format comparison tests...\n";
            test_different_formats();
            break;

        case 4:  // Базовый тест
            std::cout << "\nRunning basic functionality test...\n";
            test_basic();
            break;

        case 5:  // Сравнение 2-х моделей на одном фильтре
            std::cout << "\nComparing models test...\n";
            test_model_comparison();
            break;

        case 6:  // Пользовательский тест
        {
            std::cout << "\nRunning custom test...\n";
            data_generator::SimulationConfig config;

            // Настройка параметров
            std::cout << "Enter total steps: ";
            std::cin >> config.total_steps;

            std::cout << "Enter base dt: ";
            std::cin >> config.base_dt;

            std::cout << "Select scenario (0=ZERO_HOLD, 1=STEP, 2=SINE, 3=PULSE): ";
            int scenario;
            std::cin >> scenario;
            config.scenario.scenario2 = static_cast<model2::ControlScenario>(scenario);

            std::cout << "Select format (0=BINARY, 1=CSV, 2=MATLAB, 3=TXT): ";
            int format;
            std::cin >> format;
            config.format = static_cast<data_generator::DataFormat>(format);

            config.output_dir = "./data/custom_test";

            run_test("Custom Test", config, 88888);
        }
            break;

        default:
            std::cout << "Invalid choice. Running basic test...\n";
            test_basic();
            break;
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

    return 0;
}