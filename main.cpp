//#include "data_generator.hpp"
//#include "analysis_tools.hpp"
//#include "txt_reader.hpp"
//#include <iostream>
//#include <fstream>
//#include <iomanip>
//#include <Eigen/Dense>
//
//// Структура для хранения метрик
//struct FilterMetrics {
//    double avg_error = 0.0;
//    double max_error = 0.0;
//    double rms_error = 0.0;
//    double cov_norm = 0.0;      // норма ковариации
//    double cond_number = 0.0;   // число обусловленности
//    double symmetry_error = 0.0; // асимметрия ковариации
//};
//
//// Функция для анализа ошибок по аналогии со статьей
//void analyze_verhaegen_style(const data_generator::SimulationData& data) {
//    std::cout << "\n=== ANALYSIS IN VERHAEGEN & VAN DOOREN STYLE ===\n";
//
//    // 1. Анализ численной устойчивости
//
//    // Проверка симметрии ковариационных матриц для CKF
//    double max_ckf_asymmetry = 0.0;
//    double max_srcf_asymmetry = 0.0;
//
//    for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
//        // Асимметрия: ||P - P^T|| / ||P||
//        Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] - data.ckf_covariances[i].transpose();
//        Eigen::Matrix2d asym_srcf = data.srcf_covariances[i] - data.srcf_covariances[i].transpose();
//
//        double asym_ckf_norm = asym_ckf.norm() / data.ckf_covariances[i].norm();
//        double asym_srcf_norm = asym_srcf.norm() / data.srcf_covariances[i].norm();
//
//        max_ckf_asymmetry = std::max(max_ckf_asymmetry, asym_ckf_norm);
//        max_srcf_asymmetry = std::max(max_srcf_asymmetry, asym_srcf_norm);
//    }
//
//    std::cout << "Symmetry Analysis:\n";
//    std::cout << "  CKF maximum asymmetry: " << max_ckf_asymmetry << "\n";
//    std::cout << "  SRCF maximum asymmetry: " << max_srcf_asymmetry << "\n";
//
//    // 2. Анализ положительной определенности
//    int ckf_non_positive = 0;
//    int srcf_non_positive = 0;
//
//    for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
//        Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
//        Eigen::LLT<Eigen::Matrix2d> llt_srcf(data.srcf_covariances[i]);
//
//        if (llt_ckf.info() != Eigen::Success) ckf_non_positive++;
//        if (llt_srcf.info() != Eigen::Success) srcf_non_positive++;
//    }
//
//    std::cout << "\nPositive Definiteness:\n";
//    std::cout << "  CKF non-positive definite: " << ckf_non_positive
//              << "/" << data.ckf_covariances.size() << "\n";
//    std::cout << "  SRCF non-positive definite: " << srcf_non_positive
//              << "/" << data.srcf_covariances.size() << "\n";
//
//    // 3. Анализ ошибок фильтрации (как в статье)
//    std::vector<double> ckf_errors;
//    std::vector<double> srcf_errors;
//    std::vector<double> innovation_norms;
//
//    for (size_t i = 0; i < data.true_states.size(); ++i) {
//        // Ошибка оценки состояния
//        double ckf_err = (data.true_states[i] - data.ckf_estimates[i]).norm();
//        double srcf_err = (data.true_states[i] - data.srcf_estimates[i]).norm();
//
//        ckf_errors.push_back(ckf_err);
//        srcf_errors.push_back(srcf_err);
//
//        // Нормы инноваций (важный показатель в статье)
//        if (i > 0) {
//            Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
//            innovation_norms.push_back(innov.norm());
//        }
//    }
//
//    // 4. Вычисление метрик как в статье
//    FilterMetrics ckf_metrics, srcf_metrics;
//
//    // Средняя ошибка
//    ckf_metrics.avg_error = std::accumulate(ckf_errors.begin(), ckf_errors.end(), 0.0) / ckf_errors.size();
//    srcf_metrics.avg_error = std::accumulate(srcf_errors.begin(), srcf_errors.end(), 0.0) / srcf_errors.size();
//
//    // Максимальная ошибка
//    ckf_metrics.max_error = *std::max_element(ckf_errors.begin(), ckf_errors.end());
//    srcf_metrics.max_error = *std::max_element(srcf_errors.begin(), srcf_errors.end());
//
//    // RMS ошибка
//    double ckf_sq_sum = 0.0, srcf_sq_sum = 0.0;
//    for (size_t i = 0; i < ckf_errors.size(); ++i) {
//        ckf_sq_sum += ckf_errors[i] * ckf_errors[i];
//        srcf_sq_sum += srcf_errors[i] * srcf_errors[i];
//    }
//    ckf_metrics.rms_error = sqrt(ckf_sq_sum / ckf_errors.size());
//    srcf_metrics.rms_error = sqrt(srcf_sq_sum / srcf_errors.size());
//
//    // Нормы ковариаций
//    double ckf_cov_norm_sum = 0.0, srcf_cov_norm_sum = 0.0;
//    for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
//        ckf_cov_norm_sum += data.ckf_covariances[i].norm();
//        srcf_cov_norm_sum += data.srcf_covariances[i].norm();
//    }
//    ckf_metrics.cov_norm = ckf_cov_norm_sum / data.ckf_covariances.size();
//    srcf_metrics.cov_norm = srcf_cov_norm_sum / data.srcf_covariances.size();
//
//    // Числа обусловленности (важный параметр из статьи)
//    double ckf_cond_sum = 0.0, srcf_cond_sum = 0.0;
//    for (size_t i = 0; i < data.ckf_covariances.size(); ++i) {
//        Eigen::JacobiSVD<Eigen::Matrix2d> svd_ckf(data.ckf_covariances[i]);
//        Eigen::JacobiSVD<Eigen::Matrix2d> svd_srcf(data.srcf_covariances[i]);
//
//        double cond_ckf = svd_ckf.singularValues()(0) / svd_ckf.singularValues()(svd_ckf.singularValues().size()-1);
//        double cond_srcf = svd_srcf.singularValues()(0) / svd_srcf.singularValues()(svd_srcf.singularValues().size()-1);
//
//        ckf_cond_sum += cond_ckf;
//        srcf_cond_sum += cond_srcf;
//    }
//    ckf_metrics.cond_number = ckf_cond_sum / data.ckf_covariances.size();
//    srcf_metrics.cond_number = srcf_cond_sum / data.srcf_covariances.size();
//
//    // 5. Вывод результатов в стиле статьи
//    std::cout << "\n=== PERFORMANCE METRICS ===\n";
//    std::cout << std::fixed << std::setprecision(6);
//
//    std::cout << "\nCKF Metrics:\n";
//    std::cout << "  Average error:  " << ckf_metrics.avg_error << "\n";
//    std::cout << "  Maximum error:  " << ckf_metrics.max_error << "\n";
//    std::cout << "  RMS error:      " << ckf_metrics.rms_error << "\n";
//    std::cout << "  Avg cov norm:   " << ckf_metrics.cov_norm << "\n";
//    std::cout << "  Avg cond num:   " << ckf_metrics.cond_number << "\n";
//
//    std::cout << "\nSRCF Metrics:\n";
//    std::cout << "  Average error:  " << srcf_metrics.avg_error << "\n";
//    std::cout << "  Maximum error:  " << srcf_metrics.max_error << "\n";
//    std::cout << "  RMS error:      " << srcf_metrics.rms_error << "\n";
//    std::cout << "  Avg cov norm:   " << srcf_metrics.cov_norm << "\n";
//    std::cout << "  Avg cond num:   " << srcf_metrics.cond_number << "\n";
//
//    // 6. Сравнительный анализ
//    std::cout << "\n=== COMPARATIVE ANALYSIS ===\n";
//    std::cout << "SRCF/CKF error ratio:      " << srcf_metrics.avg_error / ckf_metrics.avg_error << "\n";
//    std::cout << "SRCF/CKF RMS ratio:        " << srcf_metrics.rms_error / ckf_metrics.rms_error << "\n";
//    std::cout << "SRCF/CKF condition ratio:  " << srcf_metrics.cond_number / ckf_metrics.cond_number << "\n";
//
//    // 7. Рекомендации как в статье
//    std::cout << "\n=== RECOMMENDATIONS (from Verhaegen & Van Dooren) ===\n";
//    if (max_ckf_asymmetry > 1e-10) {
//        std::cout << "WARNING: CKF shows significant asymmetry (" << max_ckf_asymmetry << ")\n";
//        std::cout << "  Recommendation: Use Joseph stabilized form or symmetrize P at each step\n";
//    }
//
//    if (ckf_non_positive > 0) {
//        std::cout << "WARNING: CKF produced non-positive definite covariance matrices\n";
//        std::cout << "  Recommendation: Consider SRCF for guaranteed positive definiteness\n";
//    }
//
//    if (srcf_metrics.cond_number > 1e6) {
//        std::cout << "WARNING: High condition number in SRCF (" << srcf_metrics.cond_number << ")\n";
//        std::cout << "  Recommendation: Check measurement scaling or use sequential processing\n";
//    }
//
//    // Сохранение данных для графика
//    std::ofstream error_file("verhaegen_analysis.csv");
//    error_file << "Step,CKF_Error,SRCF_Error,CKF_Cov,SRCF_Cov,Innovation\n";
//
//    for (size_t i = 0; i < data.true_states.size(); ++i) {
//        error_file << i << ","
//                   << ckf_errors[i] << ","
//                   << srcf_errors[i] << ","
//                   << data.ckf_covariances[i].norm() << ","
//                   << data.srcf_covariances[i].norm() << ",";
//
//        if (i > 0) {
//            Eigen::Vector2d innov = data.noisy_measurements[i] - data.measurements[i];
//            error_file << innov.norm();
//        } else {
//            error_file << "0";
//        }
//        error_file << "\n";
//    }
//    error_file.close();
//
//    std::cout << "\nData saved to: verhaegen_analysis.csv\n";
//}
//
//// Тест с различными уровнями шума (как в статье Test 1-6)
//void run_verhaegen_tests() {
//    std::cout << "=== RUNNING VERHAEGEN-STYLE TESTS ===\n";
//
//    std::vector<std::pair<double, double>> noise_levels = {
//            {0.1, 0.1},     // Test 1: Low noise
//            {1.0, 1.0},     // Test 2: Moderate noise
//            {10.0, 10.0},   // Test 3: High noise
//            {0.01, 1.0},    // Test 4: Low process, high measurement noise
//            {1.0, 0.01},    // Test 5: High process, low measurement noise
//    };
//
//    std::ofstream summary_file("verhaegen_summary.csv");
//    summary_file << "Test,ProcessNoise,MeasNoise,CKF_AvgError,SRCF_AvgError,CKF_RMS,SRCF_RMS,"
//                 << "CKF_Asymmetry,SRCF_Asymmetry,CKF_PosDef,SRCF_PosDef\n";
//
//    for (int test_num = 0; test_num < noise_levels.size(); ++test_num) {
//        auto [process_scale, meas_scale] = noise_levels[test_num];
//
//        std::cout << "\n--- Test " << test_num + 1 << " ---\n";
//        std::cout << "Process noise scale: " << process_scale << "\n";
//        std::cout << "Measurement noise scale: " << meas_scale << "\n";
//
//        data_generator::SimulationConfig config;
//        config.total_steps = 2000;
//        config.base_dt = 0.01;
//        config.add_process_noise = true;
//        config.add_measurement_noise = true;
//        config.process_noise_scale = process_scale;
//        config.measurement_noise_scale = meas_scale;
//        config.scenario = model2::ControlScenario::SINE_WAVE;
//        config.time_mode = time_generator::TimeMode::UNIFORM;
//        config.format = data_generator::DataFormat::BINARY;
//        config.output_dir = "./data/verhaegen_test_" + std::to_string(test_num + 1);
//
//        data_generator::DataGenerator gen(config, 12345 + test_num);
//        auto data = gen.generate();
//
//        // Анализ для этого теста
//        double ckf_avg = 0.0, srcf_avg = 0.0;
//        double ckf_rms = 0.0, srcf_rms = 0.0;
//        double ckf_asym = 0.0, srcf_asym = 0.0;
//        int ckf_posdef = 0, srcf_posdef = 0;
//
//        for (size_t i = 0; i < data.true_states.size(); ++i) {
//            // Ошибки
//            double ckf_err = (data.true_states[i] - data.ckf_estimates[i]).norm();
//            double srcf_err = (data.true_states[i] - data.srcf_estimates[i]).norm();
//
//            ckf_avg += ckf_err;
//            srcf_avg += srcf_err;
//            ckf_rms += ckf_err * ckf_err;
//            srcf_rms += srcf_err * srcf_err;
//
//            // Асимметрия
//            Eigen::Matrix2d asym_ckf = data.ckf_covariances[i] - data.ckf_covariances[i].transpose();
//            Eigen::Matrix2d asym_srcf = data.srcf_covariances[i] - data.srcf_covariances[i].transpose();
//
//            ckf_asym = std::max(ckf_asym, asym_ckf.norm() / data.ckf_covariances[i].norm());
//            srcf_asym = std::max(srcf_asym, asym_srcf.norm() / data.srcf_covariances[i].norm());
//
//            // Положительная определенность
//            Eigen::LLT<Eigen::Matrix2d> llt_ckf(data.ckf_covariances[i]);
//            Eigen::LLT<Eigen::Matrix2d> llt_srcf(data.srcf_covariances[i]);
//
//            if (llt_ckf.info() == Eigen::Success) ckf_posdef++;
//            if (llt_srcf.info() == Eigen::Success) srcf_posdef++;
//        }
//
//        ckf_avg /= data.true_states.size();
//        srcf_avg /= data.true_states.size();
//        ckf_rms = sqrt(ckf_rms / data.true_states.size());
//        srcf_rms = sqrt(srcf_rms / data.true_states.size());
//
//        summary_file << test_num + 1 << ","
//                     << process_scale << "," << meas_scale << ","
//                     << ckf_avg << "," << srcf_avg << ","
//                     << ckf_rms << "," << srcf_rms << ","
//                     << ckf_asym << "," << srcf_asym << ","
//                     << ckf_posdef << "," << srcf_posdef << "\n";
//
//        std::cout << "CKF avg error: " << ckf_avg << ", SRCF avg error: " << srcf_avg << "\n";
//        std::cout << "Ratio (SRCF/CKF): " << srcf_avg / ckf_avg << "\n";
//    }
//
//    summary_file.close();
//    std::cout << "\nSummary saved to: verhaegen_summary.csv\n";
//}
//
//void test_txt_format() {
//    std::cout << "\n=== Testing TXT Format Generation ===\n";
//
//    data_generator::SimulationConfig config;
//    config.total_steps = 500;
//    config.base_dt = 0.02;
//    config.add_process_noise = true;
//    config.add_measurement_noise = true;
//    config.scenario = model2::ControlScenario::SINE_WAVE;
//    config.format = data_generator::DataFormat::TEXT_TXT;
//    config.output_dir = "./data/txt_test";
//
//    // Генерация данных
//    data_generator::DataGenerator gen(config, 999);
//    auto sim_data = gen.generate();
//    gen.save(sim_data);
//
//    std::cout << "TXT data saved to: " << config.output_dir << "\n";
//
//    // Чтение и проверка данных
//    try {
//        auto read_data = txt_reader::readMainData(config.output_dir + "/main_data.txt");
//        txt_reader::printDataSummary(read_data);
//
//        // Проверка целостности данных
//        if (read_data.times.size() == sim_data.times.size() - 1) {
//            std::cout << "Data integrity: OK\n";
//        } else {
//            std::cout << "Data integrity: ERROR (size mismatch)\n";
//        }
//    } catch (const std::exception& e) {
//        std::cout << "Error reading TXT data: " << e.what() << "\n";
//    }
//}
//
//void generate_all_formats() {
//    std::cout << "\n=== Generating Data in All Formats ===\n";
//
//    data_generator::SimulationConfig base_config;
//    base_config.total_steps = 1000;
//    base_config.base_dt = 0.01;
//    base_config.scenario = model2::ControlScenario::STEP_MANEUVER;
//
//    // Создаем корневую директорию
//    std::string root_dir = "./data/all_formats";
//
//    // Кроссплатформенное создание директории
//#ifdef _WIN32
//    std::string command = "mkdir \"" + root_dir + "\"";
//#else
//    std::string command = "mkdir -p \"" + root_dir + "\"";
//#endif
//    system(command.c_str());
//
//    // Массив всех форматов
//    std::vector<std::pair<data_generator::DataFormat, std::string>> formats = {
//            {data_generator::DataFormat::BINARY, "binary"},
//            {data_generator::DataFormat::TEXT_CSV, "csv"},
//            {data_generator::DataFormat::TEXT_MATLAB, "matlab"},
//            {data_generator::DataFormat::TEXT_TXT, "txt"}
//    };
//
//    for (const auto& [format, name] : formats) {
//        auto config = base_config;
//        config.format = format;
//        config.output_dir = root_dir + "/" + name;
//
//        std::cout << "\nGenerating " << name << " format...\n";
//
//        // Кроссплатформенное создание поддиректории
//#ifdef _WIN32
//        command = "mkdir \"" + config.output_dir + "\" 2>nul";
//#else
//        command = "mkdir -p \"" + config.output_dir + "\"";
//#endif
//        system(command.c_str());
//
//        try {
//            data_generator::DataGenerator gen(config, 12345);
//            auto data = gen.generate();
//            gen.save(data);
//
//            std::cout << "  Saved to: " << config.output_dir << "\n";
//            std::cout << "  Files created:\n";
//
//            // Кроссплатформенный вывод содержимого директории
//#ifdef _WIN32
//            command = "dir \"" + config.output_dir + "\" /B";
//#else
//            command = "ls \"" + config.output_dir + "\"";
//#endif
//
//            std::cout << "  Executing: " << command << "\n";
//            system(command.c_str());
//
//        } catch (const std::exception& e) {
//            std::cout << "  ERROR: " << e.what() << "\n";
//        }
//    }
//
//    std::cout << "\n=== Comparison of Formats ===\n";
//    std::cout << "Binary:   Fastest, smallest files, not human-readable\n";
//    std::cout << "CSV:      Good for Excel/Spreadsheet analysis\n";
//    std::cout << "MATLAB:   Ready for MATLAB/Octave import\n";
//    std::cout << "TXT:      Human-readable, good for debugging\n";
//}
//
//void run_comprehensive_tests() {
//    // Генерация данных с разными настройками
//    std::vector<data_generator::SimulationConfig> test_configs;
//
//    // Тест 1: Разные уровни шума
//    for (double noise_scale : {0.1, 1.0, 10.0}) {
//        data_generator::SimulationConfig config;
//        config.total_steps = 2000;
//        config.measurement_noise_scale = noise_scale;
//        config.output_dir = "./data/noise_test_" + std::to_string(noise_scale);
//        test_configs.push_back(config);
//    }
//
//    // Тест 2: Разные сценарии управления
//    std::vector<model2::ControlScenario> scenarios = {
//            model2::ControlScenario::ZERO_HOLD,
//            model2::ControlScenario::STEP_MANEUVER,
//            model2::ControlScenario::SINE_WAVE,
//            model2::ControlScenario::PULSE
//    };
//
//    for (auto scenario : scenarios) {
//        data_generator::SimulationConfig config;
//        config.total_steps = 2000;
//        config.scenario = scenario;
//        config.output_dir = "./data/scenario_" + std::to_string(static_cast<int>(scenario));
//        test_configs.push_back(config);
//    }
//
//    // Запуск всех тестов
//    for (const auto& config : test_configs) {
//        data_generator::DataGenerator gen(config);
//        auto data = gen.generate();
//        gen.save(data);
//
//        // Анализ результатов
//        std::string report_file = config.output_dir + "/performance_report.txt";
//        analysis::generate_comparison_report(data, report_file);
//
//        std::cout << "Test completed: " << config.output_dir << "\n";
//    }
//}
//
//void generate_test_data() {
//    std::cout << "\n=== Generating Test Data ===\n";
//
//    // Конфигурация 1: Бинарные данные для быстрых тестов
//    {
//        data_generator::SimulationConfig config;
//        config.total_steps = 5000;
//        config.base_dt = 0.01;
//        config.add_process_noise = true;
//        config.add_measurement_noise = true;
//        config.process_noise_scale = 1.0;
//        config.measurement_noise_scale = 1.0;
//        config.scenario = model2::ControlScenario::SINE_WAVE;
//        config.time_mode = time_generator::TimeMode::RANDOM_JITTER;
//        config.format = data_generator::DataFormat::BINARY;
//        config.output_dir = "./data/binary";
//
//        data_generator::DataGenerator gen(config, 12345);
//        auto data = gen.generate();
//        gen.save(data);
//
//        std::cout << "Binary data saved to: " << config.output_dir << "\n";
//    }
//
//    // Конфигурация 2: CSV данные для анализа
//    {
//        data_generator::SimulationConfig config;
//        config.total_steps = 1000;
//        config.base_dt = 0.01;
//        config.add_process_noise = true;
//        config.add_measurement_noise = true;
//        config.scenario = model2::ControlScenario::STEP_MANEUVER;
//        config.time_mode = time_generator::TimeMode::UNIFORM;
//        config.format = data_generator::DataFormat::TEXT_CSV;
//        config.output_dir = "./data/csv";
//
//        data_generator::DataGenerator gen(config, 54321);
//        auto data = gen.generate();
//        gen.save(data);
//
//        std::cout << "CSV data saved to: " << config.output_dir << "\n";
//    }
//
//    // Конфигурация 3: MATLAB данные
//    {
//        data_generator::SimulationConfig config;
//        config.total_steps = 2000;
//        config.base_dt = 0.005; // Более высокая частота
//        config.add_process_noise = false; // Без шума для анализа
//        config.add_measurement_noise = true;
//        config.scenario = model2::ControlScenario::PULSE;
//        config.format = data_generator::DataFormat::TEXT_MATLAB;
//        config.output_dir = "./data/matlab";
//
//        data_generator::DataGenerator gen(config, 98765);
//        auto data = gen.generate();
//        gen.save(data);
//
//        std::cout << "MATLAB data saved to: " << config.output_dir << "\n";
//    }
//}
//
//void compare_filters_with_data() {
//    std::cout << "\n=== Comparing Filters with Generated Data ===\n";
//
//    data_generator::SimulationConfig config;
//    config.total_steps = 3000;
//    config.base_dt = 0.01;
//    config.add_process_noise = true;
//    config.add_measurement_noise = true;
//    config.scenario = model2::ControlScenario::SINE_WAVE;
//    config.format = data_generator::DataFormat::BINARY;
//    config.output_dir = "./data/comparison";
//
//    data_generator::DataGenerator gen(config, 11111);
//    auto data = gen.generate();
//    gen.save(data);
//
//    // Анализ ошибок
//    double ckf_total_error = 0.0;
//    double srcf_total_error = 0.0;
//
//    for (size_t i = 0; i < data.true_states.size(); ++i) {
//        double ckf_error = (data.true_states[i] - data.ckf_estimates[i]).norm();
//        double srcf_error = (data.true_states[i] - data.srcf_estimates[i]).norm();
//
//        ckf_total_error += ckf_error;
//        srcf_total_error += srcf_error;
//    }
//
//    double ckf_avg_error = ckf_total_error / data.true_states.size();
//    double srcf_avg_error = srcf_total_error / data.true_states.size();
//
//    std::cout << "\n=== Error Analysis ===\n";
//    std::cout << "CKF average error: " << ckf_avg_error << "\n";
//    std::cout << "SRCF average error: " << srcf_avg_error << "\n";
//    std::cout << "Ratio (SRCF/CKF): " << srcf_avg_error / ckf_avg_error << "\n";
//
//    if (srcf_avg_error > ckf_avg_error * 1.5) {
//        std::cout << "WARNING: SRCF significantly worse than CKF!\n";
//    } else if (srcf_avg_error < ckf_avg_error) {
//        std::cout << "SUCCESS: SRCF performs better than CKF!\n";
//    } else {
//        std::cout << "INFO: Both filters have similar performance.\n";
//    }
//}
//
//int main(int argc, char* argv[]) {
//    std::cout << "=== VERHAEGEN & VAN DOOREN (1986) STYLE ANALYSIS ===\n";
//
//    // 1. Основной эксперимент
//    {
//        std::cout << "\n--- Main Experiment ---\n";
//
//        data_generator::SimulationConfig config;
//        config.total_steps = 3000;
//        config.base_dt = 0.01;
//        config.add_process_noise = true;
//        config.add_measurement_noise = true;
//        config.process_noise_scale = 1.0;
//        config.measurement_noise_scale = 1.0;
//        config.scenario = model2::ControlScenario::SINE_WAVE;
//        config.time_mode = time_generator::TimeMode::RANDOM_JITTER;
//        config.format = data_generator::DataFormat::BINARY;
//        config.output_dir = "./data/verhaegen_main";
//
//        data_generator::DataGenerator gen(config, 42);
//        auto data = gen.generate();
//
//        // Анализ в стиле статьи
//        analyze_verhaegen_style(data);
//    }
//
//    // 2. Серия тестов с разными уровнями шума
//    run_verhaegen_tests();
//
//    // 3. Тест с неустойчивой системой (как в статье)
//    {
//        std::cout << "\n--- Test with Unstable System ---\n";
//        std::cout << "Modifying system parameters to create instability...\n";
//
//        // Можно временно изменить параметры системы для теста неустойчивости
//        // Например, уменьшить демпфирование
//
//        data_generator::SimulationConfig config;
//        config.total_steps = 2000;
//        config.base_dt = 0.01;
//        config.add_process_noise = true;
//        config.add_measurement_noise = true;
//        config.process_noise_scale = 5.0;  // Увеличенный шум процесса
//        config.measurement_noise_scale = 0.1;  // Низкий шум измерений
//        config.scenario = model2::ControlScenario::STEP_MANEUVER;
//        config.format = data_generator::DataFormat::TEXT_CSV;
//        config.output_dir = "./data/unstable_test";
//
//        data_generator::DataGenerator gen(config, 99);
//        auto data = gen.generate();
//
//        // Анализ
//        analyze_verhaegen_style(data);
//    }
//    return 0;
//}
#include "data_generator.hpp"
#include "analysis_tools.hpp"
#include "txt_reader.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <filesystem>

// Структура для хранения метрик
struct FilterMetrics {
    double avg_error = 0.0;
    double max_error = 0.0;
    double rms_error = 0.0;
    double cov_norm = 0.0;      // норма ковариации
    double cond_number = 0.0;   // число обусловленности
    double symmetry_error = 0.0; // асимметрия ковариации
};

// Функция для анализа ошибок по аналогии со статьей
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

    ckf_metrics.avg_error = std::accumulate(ckf_errors.begin(), ckf_errors.end(), 0.0) / ckf_errors.size();
    srcf_metrics.avg_error = std::accumulate(srcf_errors.begin(), srcf_errors.end(), 0.0) / srcf_errors.size();

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
    std::cout << "  Average error:  " << ckf_metrics.avg_error << "\n";
    std::cout << "  Maximum error:  " << ckf_metrics.max_error << "\n";
    std::cout << "  RMS error:      " << ckf_metrics.rms_error << "\n";
    std::cout << "  Avg cov norm:   " << ckf_metrics.cov_norm << "\n";
    std::cout << "  Avg cond num:   " << ckf_metrics.cond_number << "\n";

    std::cout << "\nSRCF Metrics:\n";
    std::cout << "  Average error:  " << srcf_metrics.avg_error << "\n";
    std::cout << "  Maximum error:  " << srcf_metrics.max_error << "\n";
    std::cout << "  RMS error:      " << srcf_metrics.rms_error << "\n";
    std::cout << "  Avg cov norm:   " << srcf_metrics.cov_norm << "\n";
    std::cout << "  Avg cond num:   " << srcf_metrics.cond_number << "\n";

    std::cout << "\n=== COMPARATIVE ANALYSIS ===\n";
    std::cout << "SRCF/CKF error ratio:      " << srcf_metrics.avg_error / ckf_metrics.avg_error << "\n";
    std::cout << "SRCF/CKF RMS ratio:        " << srcf_metrics.rms_error / ckf_metrics.rms_error << "\n";
    std::cout << "SRCF/CKF condition ratio:  " << srcf_metrics.cond_number / ckf_metrics.cond_number << "\n";

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

// Тест с различными уровнями шума
void run_verhaegen_tests() {
    std::cout << "=== RUNNING VERHAEGEN-STYLE TESTS ===\n";

    std::vector<std::pair<double, double>> noise_levels = {
            {0.1, 0.1},
            {1.0, 1.0},
            {10.0, 10.0},
            {0.01, 1.0},
            {1.0, 0.01},
    };

    std::ofstream summary_file("verhaegen_summary.csv");
    summary_file << "Test,ProcessNoise,MeasNoise,CKF_AvgError,SRCF_AvgError,CKF_RMS,SRCF_RMS,"
                 << "CKF_Asymmetry,SRCF_Asymmetry,CKF_PosDef,SRCF_PosDef\n";

    for (int test_num = 0; test_num < noise_levels.size(); ++test_num) {
        auto [process_scale, meas_scale] = noise_levels[test_num];

        std::cout << "\n--- Test " << test_num + 1 << " ---\n";
        std::cout << "Process noise scale: " << process_scale << "\n";
        std::cout << "Measurement noise scale: " << meas_scale << "\n";

        data_generator::SimulationConfig config;
        config.total_steps = 2000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.process_noise_scale = process_scale;
        config.measurement_noise_scale = meas_scale;
        config.scenario = model2::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::UNIFORM;
        config.format = data_generator::DataFormat::BINARY; // Только BINARY для тестов
        config.output_dir = "./data/verhaegen_test_" + std::to_string(test_num + 1);

        data_generator::DataGenerator gen(config, 12345 + test_num);
        auto data = gen.generate();
        gen.save(data);

        // Анализ для этого теста
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

        std::cout << "CKF avg error: " << ckf_avg << ", SRCF avg error: " << srcf_avg << "\n";
        std::cout << "Ratio (SRCF/CKF): " << srcf_avg / ckf_avg << "\n";
    }

    summary_file.close();
    std::cout << "\nSummary saved to: verhaegen_summary.csv\n";
}

// Генерация в двух форматах: BINARY и TEXT_TXT
void generate_two_formats() {
    std::cout << "\n=== GENERATING DATA IN TWO FORMATS ===\n";

    data_generator::SimulationConfig base_config;
    base_config.total_steps = 1000;
    base_config.base_dt = 0.01;
    base_config.add_process_noise = true;
    base_config.add_measurement_noise = true;
    base_config.process_noise_scale = 1.0;
    base_config.measurement_noise_scale = 1.0;
    base_config.scenario = model2::ControlScenario::STEP_MANEUVER;
    base_config.time_mode = time_generator::TimeMode::UNIFORM;

    // Массив двух форматов
    std::vector<std::pair<data_generator::DataFormat, std::string>> formats = {
            {data_generator::DataFormat::BINARY, "binary"},
            {data_generator::DataFormat::TEXT_TXT, "txt"}
    };

    for (const auto& [format, name] : formats) {
        auto config = base_config;
        config.format = format;
        config.output_dir = "./data/two_formats/" + name;

        // Создаем директорию
        std::filesystem::create_directories(config.output_dir);

        std::cout << "\nGenerating " << name << " format...\n";

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

        } catch (const std::exception& e) {
            std::cout << "  ✗ ERROR: " << e.what() << "\n";
        }
    }

    // Сравнение форматов
    std::cout << "\n=== FORMAT COMPARISON ===\n";

    // Сравним размеры файлов
    std::string binary_dir = "./data/two_formats/binary";
    std::string txt_dir = "./data/two_formats/txt";

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
    config.scenario = model2::ControlScenario::SINE_WAVE;
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

    // Чтение и проверка данных (если есть txt_reader)
#ifdef HAS_TXT_READER
    try {
        auto read_data = txt_reader::readMainData(config.output_dir + "/main_data.txt");
        txt_reader::printDataSummary(read_data);

        if (read_data.times.size() == sim_data.times.size() - 1) {
            std::cout << "\n✓ Data integrity: OK\n";
        } else {
            std::cout << "\n✗ Data integrity: ERROR (size mismatch)\n";
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Error reading TXT data: " << e.what() << "\n";
    }
#else
    std::cout << "\nℹ TXT reader not available (compile with txt_reader support)\n";
#endif
}

int main(int argc, char* argv[]) {
    std::cout << "=== KALMAN FILTER DATA GENERATION (Binary & TXT Only) ===\n";

    // Создаем корневую директорию
    std::filesystem::create_directories("./data");

    // 1. Генерация в двух форматах
    generate_two_formats();

    // 2. Тестирование TXT формата
    test_txt_format();

    // 3. Основной эксперимент в стиле Verhaegen & Van Dooren (только BINARY)
    {
        std::cout << "\n\n=== MAIN VERHAEGEN EXPERIMENT ===\n";

        data_generator::SimulationConfig config;
        config.total_steps = 3000;
        config.base_dt = 0.01;
        config.add_process_noise = true;
        config.add_measurement_noise = true;
        config.process_noise_scale = 1.0;
        config.measurement_noise_scale = 1.0;
        config.scenario = model2::ControlScenario::SINE_WAVE;
        config.time_mode = time_generator::TimeMode::RANDOM_JITTER;
        config.format = data_generator::DataFormat::BINARY; // Только BINARY
        config.output_dir = "./data/verhaegen_main";

        std::filesystem::create_directories(config.output_dir);

        data_generator::DataGenerator gen(config, 42);
        auto data = gen.generate();
        gen.save(data);

        analyze_verhaegen_style(data);
    }

    // 4. Серия тестов с разными уровнями шума (только BINARY)
    run_verhaegen_tests();

    std::cout << "\n=== EXPERIMENT COMPLETED ===\n";
    std::cout << "Generated data in:\n";
    std::cout << "  • ./data/two_formats/ - Binary and TXT formats\n";
    std::cout << "  • ./data/txt_test/ - TXT format test\n";
    std::cout << "  • ./data/verhaegen_main/ - Main experiment\n";
    std::cout << "  • ./data/verhaegen_test_*/ - Noise level tests\n";

    return 0;
}