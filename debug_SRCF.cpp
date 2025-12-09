//
// Created by Vladimir on 09.12.2025.
//
#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include "kalman.hpp"
#include "models.hpp"
#include <chrono>

void test_performance()
{
    using namespace model2;
    using namespace std::chrono;

    std::cout << "\n=== Performance Test ===\n";

    const int steps = 10000;
    double dt = 0.01;

    Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
    Eigen::Vector2d x_est = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman::SRCF filter(x_est, P0);

    // Замер времени
    auto start = high_resolution_clock::now();

    for (int k = 0; k < steps; ++k)
    {
        double t = k * dt;

        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();
        auto u = model2::u(t, ControlScenario::SINE_WAVE);

        x_true = A * x_true + B * u;
        Eigen::Vector2d y = C * x_true;

        filter.step(A, B, C, D, Q, R, u, y);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Time per step: " << duration.count() / (double)steps << " μs" << std::endl;
    std::cout << "Max frequency: " << 1e6 / (duration.count() / (double)steps) << " Hz" << std::endl;
}

void basic_sanity_test()
{
    using namespace Eigen;

    // A = I
    Matrix2d A = Matrix2d::Identity();

    // B = I
    Matrix2d B = Matrix2d::Identity();

    // C = I
    Matrix2d C = Matrix2d::Identity();

    // D=0
    Matrix2d D = Matrix2d::Zero();

    // No process noise
    Matrix2d Q = Matrix2d::Zero();

    // No measurement noise
    Matrix2d R = Matrix2d::Zero();

    // Initial state
    Vector2d x0(1.0, 2.0);
    kalman::SRCF filter(x0, Matrix2d::Identity()*1e-3);

    Vector2d x = x0;

    std::cout << "=== TEST A=I, B=I, C=I, no noise ===\n";
    for (int k=0;k<5;k++)
    {
        Vector2d u(1.0,0.5);
        Vector2d y = x;  // measurement matches truth

        // TRUE update
        x = A*x + B*u;

        // KF update
        filter.step(A,B,C,D,Q,R,u,y);

        std::cout << "k="<<k
                  << " true="<<x.transpose()
                  << " est ="<<filter.state().transpose()
                  << std::endl;
    }
}

void test_SRCF_manual_step(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                           const Eigen::VectorXd& u, const Eigen::VectorXd& y,
                           kalman::SRCF& filter) {

    Eigen::VectorXd x = filter.state();
    Eigen::MatrixXd S = filter.covarianceSqrt();

    std::cout << "\n=== Manual SRCF Step ===\n";

    const size_t n = x.size();
    const size_t p = y.size();
    const size_t m = B.cols();

    std::cout << "n=" << n << ", p=" << p << ", m=" << m << "\n";

    // 1. Квадратные корни
    std::cout << "\n1. Cholesky decompositions:\n";
    Eigen::MatrixXd SQ = Q.llt().matrixL();
    Eigen::MatrixXd SR = R.llt().matrixL();
    std::cout << "SQ (" << SQ.rows() << "x" << SQ.cols() << "): " << SQ << "\n";
    std::cout << "SR (" << SR.rows() << "x" << SR.cols() << "): " << SR << "\n";

    // 2. Вычисляем блоки prearray
    std::cout << "\n2. Computing prearray blocks:\n";
    Eigen::MatrixXd C_times_S = C * S;
    Eigen::MatrixXd A_times_S = A * S;
    Eigen::MatrixXd B_times_SQ = B * SQ;

    std::cout << "C*S (" << C_times_S.rows() << "x" << C_times_S.cols() << "):\n" << C_times_S << "\n";
    std::cout << "A*S (" << A_times_S.rows() << "x" << A_times_S.cols() << "):\n" << A_times_S << "\n";
    std::cout << "B*SQ (" << B_times_SQ.rows() << "x" << B_times_SQ.cols() << "):\n" << B_times_SQ << "\n";

    // 3. Формируем prearray
    std::cout << "\n3. Building prearray (" << (p+n) << "x" << (p+n+m) << "):\n";
    Eigen::MatrixXd prearray(p + n, p + n + m);
    prearray.setZero();

    prearray.block(0, 0, p, p) = SR;
    prearray.block(0, p, p, n) = C_times_S;
    prearray.block(p, p, n, n) = A_times_S;
    prearray.block(p, p + n, n, m) = B_times_SQ;

    std::cout << "Prearray:\n" << prearray << "\n";

    // 4. QR разложение
    std::cout << "\n4. QR decomposition:\n";
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(prearray.transpose());
    Eigen::MatrixXd R_mat = qr.matrixQR().triangularView<Eigen::Upper>();

    std::cout << "R_mat (" << R_mat.rows() << "x" << R_mat.cols() << "):\n" << R_mat << "\n";

    // 5. Postarray
    Eigen::MatrixXd postarray = R_mat.topRows(p + n).transpose();
    std::cout << "\n5. Postarray (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";
    for (int i = 0; i < p+n; i++) {
        if (postarray(i,i) < 0) {
            postarray.block(0,i,p+n,1) *= -1;
        }
    }

    // 6. Извлекаем блоки
    std::cout << "\n6. Extracting blocks:\n";
    Eigen::MatrixXd S_Re = postarray.block(0, 0, p, p);
    Eigen::MatrixXd G = postarray.block(p, 0, n, p);
    Eigen::MatrixXd S_new = postarray.block(p, p, n, n);

    std::cout << "S_Re (" << S_Re.rows() << "x" << S_Re.cols() << "): " << S_Re << "\n";
    std::cout << "G (" << G.rows() << "x" << G.cols() << "):\n" << G << "\n";
    std::cout << "S_new (" << S_new.rows() << "x" << S_new.cols() << "):\n" << S_new << "\n";

    // 7. Обновление состояния
    std::cout << "\n7. State update:\n";
    Eigen::VectorXd innov = y - C * x;
    std::cout << "Innovation: " << innov.transpose() << "\n";

    Eigen::VectorXd z = S_Re.triangularView<Eigen::Lower>().solve(innov);
    std::cout << "z = S_Re\\innov: " << z.transpose() << "\n";

    Eigen::VectorXd x_new = A * x + G * z + D * u;
    std::cout << "x_new = A*x + G*z + D*u:\n";
    std::cout << "  A*x = " << (A * x).transpose() << "\n";
    std::cout << "  G*z = " << (G * z).transpose() << "\n";
    std::cout << "  D*u = " << (D * u).transpose() << "\n";
    std::cout << "  x_new = " << x_new.transpose() << "\n";

    std::cout << "\n=== Summary ===\n";
    std::cout << "Old state: " << x.transpose() << "\n";
    std::cout << "New state: " << x_new.transpose() << "\n";
    std::cout << "True state: " << (A * Eigen::Vector2d(0,0) + B * u).transpose() << "\n";
}

void debug_SRCF_step_by_step() {
    std::cout << "=== DEBUG SRCF Step-by-Step ===\n";

    const int n = 2, p = 1, m = 1;
    double dt = 0.1;

    Eigen::Matrix2d A;
    A << 1.0, dt, 0.0, 1.0;

    Eigen::MatrixXd B(2, 1);
    B << 0.5*dt*dt, dt;

    Eigen::MatrixXd C(1, 2);
    C << 1.0, 0.0;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 1);

    Eigen::MatrixXd Q(1, 1);
    Q << 0.01;

    Eigen::MatrixXd R(1, 1);
    R << 0.1;

    Eigen::Vector2d x0(0.0, 0.0);
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0, 0.0, 1.0;

    kalman::SRCF srcf(x0, P0);

    std::cout << "\n=== Initial state ===\n";
    std::cout << "x0: " << srcf.state().transpose() << "\n";
    std::cout << "S0:\n" << srcf.covarianceSqrt() << "\n";

    // ОДИН тестовый шаг с максимальной отладкой
    Eigen::VectorXd u(1);
    u << 1.0;

    Eigen::Vector2d x_true = x0;
    x_true = A * x_true + B * u;

    Eigen::VectorXd y(1);
    y << 0.1;  // измерение (примерно равно положению)

    std::cout << "\n=== Calling step() ===\n";
    std::cout << "Input y: " << y.transpose() << "\n";
    std::cout << "True x: " << x_true.transpose() << "\n";

    // Временно модифицируем SRCF::step для детальной отладки
    // Или создадим тестовую функцию
    test_SRCF_manual_step(A, B, C, D, Q, R, u, y, srcf);
}

void test_with_sign_correction() {
    std::cout << "=== TEST WITH SIGN CORRECTION ===\n";

    // Та же система
    const int n = 2, p = 1, m = 1;
    double dt = 0.1;

    Eigen::Matrix2d A;
    A << 1.0, dt, 0.0, 1.0;

    Eigen::MatrixXd B(2, 1);
    B << 0.5*dt*dt, dt;

    Eigen::MatrixXd C(1, 2);
    C << 1.0, 0.0;

    Eigen::MatrixXd Q(1, 1);
    Q << 0.01;

    Eigen::MatrixXd R(1, 1);
    R << 0.1;

    Eigen::Vector2d x0(0.0, 0.0);
    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    Eigen::VectorXd x = x0;
    Eigen::MatrixXd S = P0.llt().matrixL();

    // Вычисляем как в первоначальном SRCF
    Eigen::MatrixXd SQ = Q.llt().matrixL();
    Eigen::MatrixXd SR = R.llt().matrixL();

    Eigen::MatrixXd M(p + n, p + n + m);
    M.setZero();
    M.block(0, 0, p, p) = SR;
    M.block(0, p, p, n) = C * S;
    M.block(p, p, n, n) = A * S;
    M.block(p, p + n, n, m) = B * SQ;

    // QR без коррекции знака
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M.transpose());
    Eigen::MatrixXd R_mat = qr.matrixQR().triangularView<Eigen::Upper>();

    Eigen::MatrixXd post = R_mat.topRows(p + n).transpose();

    std::cout << "Postarray (NO sign correction):\n" << post << "\n";

    for (int i = 0; i < p+n; i++) {
        if (post(i,i) < 0) {
            post.block(0,i,p+n,1) *= -1;
        }
    }

    // Извлекаем блоки
    Eigen::MatrixXd S_Re = post.block(0, 0, p, p);
    Eigen::MatrixXd G = post.block(p, 0, n, p);
    Eigen::MatrixXd S_new = post.block(p, p, n, n);

    std::cout << "\nS_Re: " << S_Re << "\n";
    std::cout << "G:\n" << G << "\n";
    std::cout << "S_new:\n" << S_new << "\n";

    // Обновление состояния
    Eigen::VectorXd y_vec(1); y_vec << 0.1;
    Eigen::VectorXd innov = y_vec - C * x;

    // Вариант 1: как в первоначальном коде
    Eigen::VectorXd x1 = A * x + G * S_Re.inverse() * innov;

    // Вариант 2: через решение треугольной системы
    Eigen::VectorXd z = S_Re.triangularView<Eigen::Lower>().solve(innov);
    Eigen::VectorXd x2 = A * x + G * z;

    std::cout << "\nUpdate with G * S_Re^-1 * innov: " << x1.transpose() << "\n";
    std::cout << "Update with G * z (z = S_Re\\innov): " << x2.transpose() << "\n";

    // CKF для сравнения
    Eigen::Matrix2d P = S * S.transpose();
    Eigen::Matrix2d P_pred = A * P * A.transpose() + B * Q * B.transpose();
    Eigen::MatrixXd K = P_pred * C.transpose() * (C * P_pred * C.transpose() + R).inverse();
    Eigen::Vector2d x_ckf = A * x + K * (y_vec - C * A * x);

    std::cout << "\nCKF result: " << x_ckf.transpose() << "\n";
    std::cout << "Difference SRCF1-CKF: " << (x1 - x_ckf).norm() << "\n";
    std::cout << "Difference SRCF2-CKF: " << (x2 - x_ckf).norm() << "\n";
}

void test_model2_simple()
{
    using namespace model2;

    std::cout << "\n=== Test model2 simple ===\n";

    double T0 = 0.0;
    double dt = 0.01;
    const int steps = 2000;

    auto A_mat = A(dt);
    auto B_mat = B(dt);
    auto C_mat = C();
    auto Q_mat = Q(dt);
    auto R_mat = R();

    std::cout << "A: " << A_mat.rows() << "x" << A_mat.cols() << std::endl;
    std::cout << "B: " << B_mat.rows() << "x" << B_mat.cols() << std::endl;
    std::cout << "C: " << C_mat.rows() << "x" << C_mat.cols() << std::endl;
    std::cout << "Q: " << Q_mat.rows() << "x" << Q_mat.cols() << std::endl;
    std::cout << "R: " << R_mat.rows() << "x" << R_mat.cols() << std::endl;

    Eigen::Vector2d x = Eigen::Vector2d::Zero(); // exact
    Eigen::Vector2d x_f = Eigen::Vector2d::Zero(); // filter

    Eigen::Matrix2d P0;
    P0 << 1,0, 0,1;

    kalman::SRCF f(x_f, P0);

    for (int k=0; k<steps; ++k)
    {
        double t = T0 + k*dt;

        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();

        auto u = model2::u(t, model2::ControlScenario::SINE_WAVE);

        // true
        x = A*x + B*u;  // no process noise first test

        // ideal measurement
        Eigen::Vector2d y = C*x;

        // filter step
        f.step(A, B, C, D, Q, R, u, y);

        if (k % 200 == 0)
        {
            std::cout << "t=" << t
                      << "  φ=" << x(0)
                      << "  φ_f=" << f.state()(0)
                      << "  p=" << x(1)
                      << "  p_f=" << f.state()(1)
                      << "\n";
        }
    }

    std::cout << "=== End ===\n";
}

void test_model2_with_metrics()
{
    using namespace model2;

    std::cout << "\n=== Test model2 with metrics ===\n";

    double T0 = 0.0;
    double dt = 0.01;
    const int steps = 2000;

    // Инициализация
    Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
    Eigen::Vector2d x_est = Eigen::Vector2d::Zero();

    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0,
            0.0, 1.0;

    kalman::SRCF filter(x_est, P0);

    // Статистика
    double sum_sq_error_phi = 0.0;
    double sum_sq_error_p = 0.0;

    for (int k = 0; k < steps; ++k)
    {
        double t = T0 + k * dt;

        // Матрицы системы
        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();

        // Управление
        auto u = model2::u(t, model2::ControlScenario::SINE_WAVE);

        // Истинное состояние (без шума)
        x_true = A * x_true + B * u;

        // Идеальные измерения (без шума)
        Eigen::Vector2d y = C * x_true;

        // Шаг фильтра
        filter.step(A, B, C, D, Q, R, u, y);

        // Ошибки
        double error_phi = std::abs(x_true(0) - filter.state()(0));
        double error_p = std::abs(x_true(1) - filter.state()(1));

        sum_sq_error_phi += error_phi * error_phi;
        sum_sq_error_p += error_p * error_p;

        // Вывод каждые 200 шагов
        if (k % 200 == 0)
        {
            std::cout << "t=" << std::fixed << std::setprecision(2) << t
                      << " | φ: true=" << std::setw(8) << std::setprecision(4) << x_true(0)
                      << ", est=" << std::setw(8) << filter.state()(0)
                      << ", err=" << std::setw(8) << error_phi
                      << " | p: true=" << std::setw(8) << x_true(1)
                      << ", est=" << std::setw(8) << filter.state()(1)
                      << ", err=" << std::setw(8) << error_p
                      << std::endl;
        }
    }

    // Вычисление метрик
    double rms_phi = std::sqrt(sum_sq_error_phi / steps);
    double rms_p = std::sqrt(sum_sq_error_p / steps);

    // ARMSE (Average Root Mean Square Error)
    double armse = std::sqrt((rms_phi * rms_phi + rms_p * rms_p) / 2.0);

    // Итоговая статистика
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RMS error φ: " << rms_phi << " rad" << std::endl;
    std::cout << "RMS error p: " << rms_p << " rad/s" << std::endl;
    std::cout << "ARMSE (φ & p): " << armse << std::endl;

    // Дополнительные метрики - САМАЯ ПРОСТАЯ ВЕРСИЯ
    std::cout << "\n=== Additional Metrics ===" << std::endl;

    try {
        Eigen::Matrix2d P_final = filter.covariance();
        std::cout << "Final covariance matrix P:\n" << P_final << std::endl;

        // Просто детерминант и след
        double det = P_final.determinant();
        double trace = P_final.trace();
        std::cout << "Determinant of P: " << det << std::endl;
        std::cout << "Trace of P: " << trace << std::endl;

        // Простая проверка: все ли диагональные элементы положительны
        if (P_final(0,0) > 0 && P_final(1,1) > 0) {
            std::cout << "P diagonal elements positive ✓" << std::endl;
        } else {
            std::cout << "P has non-positive diagonal elements ✗" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error computing covariance metrics: " << e.what() << std::endl;
    }

    std::cout << "=== End ===" << std::endl;
}

void test_model2_with_measurement_noise()
{
    using namespace model2;

    std::cout << "\n=== Test with measurement noise ===\n";

    double dt = 0.01;
    const int steps = 2000;

    // Сброс генератора шума
    model2::reset_noise();

    Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
    Eigen::Vector2d x_est = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman::SRCF filter(x_est, P0);

    for (int k = 0; k < steps; ++k)
    {
        double t = k * dt;

        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();
        auto u = model2::u(t, model2::ControlScenario::SINE_WAVE);

        // Истинное состояние
        x_true = A * x_true + B * u;

        // Измерения с шумом
        Eigen::Vector2d y = C * x_true + model2::v(t, true);

        // Шаг фильтра
        filter.step(A, B, C, D, Q, R, u, y);

        if (k % 200 == 0)
        {
            std::cout << "t=" << t
                      << " φ_err=" << std::abs(x_true(0) - filter.state()(0))
                      << " p_err=" << std::abs(x_true(1) - filter.state()(1))
                      << std::endl;
        }
    }
}

void test_model2_with_process_noise()
{
    using namespace model2;

    std::cout << "\n=== Test with process noise ===\n";

    double dt = 0.01;
    const int steps = 2000;

    model2::reset_noise();  // Сброс коррелированного шума

    Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
    Eigen::Vector2d x_est = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman::SRCF filter(x_est, P0);

    for (int k = 0; k < steps; ++k)
    {
        double t = k * dt;

        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();
        auto u = model2::u(t, model2::ControlScenario::SINE_WAVE);

        // Истинное состояние с шумом процесса
        x_true = A * x_true + B * u + model2::w(t, dt, true);

        // Измерения (без шума для чистоты теста)
        Eigen::Vector2d y = C * x_true;

        // Шаг фильтра
        filter.step(A, B, C, D, Q, R, u, y);

        if (k % 200 == 0)
        {
            std::cout << "t=" << t
                      << " φ_err=" << std::abs(x_true(0) - filter.state()(0))
                      << " p_err=" << std::abs(x_true(1) - filter.state()(1))
                      << std::endl;
        }
    }
}

void test_model2_different_scenarios()
{
    using namespace model2;

    std::vector<ControlScenario> scenarios = {
            ControlScenario::ZERO_HOLD,
            ControlScenario::STEP_MANEUVER,
            ControlScenario::SINE_WAVE,
            ControlScenario::PULSE
    };

    std::vector<std::string> scenario_names = {
            "Zero Hold (Autopilot)",
            "Step Maneuver",
            "Sine Wave",
            "Pulse"
    };

    for (size_t i = 0; i < scenarios.size(); ++i)
    {
        std::cout << "\n=== Scenario: " << scenario_names[i] << " ===\n";

        double dt = 0.01;
        const int steps = 1000;

        Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
        Eigen::Vector2d x_est = Eigen::Vector2d::Zero();
        Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

        kalman::SRCF filter(x_est, P0);

        double total_error = 0.0;

        for (int k = 0; k < steps; ++k)
        {
            double t = k * dt;

            auto A = model2::A(dt);
            auto B = model2::B(dt);
            auto C = model2::C();
            auto D = model2::D();
            auto Q = model2::Q(dt);
            auto R = model2::R();
            auto u = model2::u(t, scenarios[i]);

            // Истинное состояние
            x_true = A * x_true + B * u;

            // Измерения
            Eigen::Vector2d y = C * x_true;

            // Шаг фильтра
            filter.step(A, B, C, D, Q, R, u, y);

            total_error += (x_true - filter.state()).norm();
        }

        std::cout << "Average error: " << total_error / steps << std::endl;
        std::cout << "Final state: φ=" << filter.state()(0)
                  << ", p=" << filter.state()(1) << std::endl;
    }
}

void comprehensive_comparison_SRCF_vs_CKF()
{
    using namespace model2;
    using namespace std::chrono;

    std::cout << "\n=== Comprehensive Comparison: SRCF vs CKF ===\n";
    std::cout << "Based on Verhaegen & Van Dooren (1986) analysis\n";

    // Параметры теста
    const int num_tests = 5;
    const int steps_per_test = 1000;
    double dt = 0.01;

    struct FilterStats {
        double total_error_phi = 0.0;
        double total_error_p = 0.0;
        double total_time = 0.0; // микросекунды
        double max_error = 0.0;
        double condition_number = 0.0;
        bool positive_definite = true;
    };

    FilterStats srcf_stats, ckf_stats;

    // Для хранения ошибок по времени
    std::vector<std::vector<double>> srcf_errors_phi(num_tests);
    std::vector<std::vector<double>> srcf_errors_p(num_tests);
    std::vector<std::vector<double>> ckf_errors_phi(num_tests);
    std::vector<std::vector<double>> ckf_errors_p(num_tests);

    for (int test = 0; test < num_tests; ++test) {
        std::cout << "\n--- Test " << test + 1 << "/" << num_tests << " ---\n";

        // Случайные начальные условия
        Eigen::Vector2d x_true = Eigen::Vector2d::Random() * 3.0;
        Eigen::Vector2d x_srcf = x_true + Eigen::Vector2d::Random() * 1.0;
        Eigen::Vector2d x_ckf = x_srcf;

        Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity() * 5.0;

        // Инициализация фильтров
        kalman::SRCF srcf_filter(x_srcf, P0);
        kalman::CKF ckf_filter(x_ckf, P0);

        // Ресайз векторов ошибок
        srcf_errors_phi[test].reserve(steps_per_test);
        srcf_errors_p[test].reserve(steps_per_test);
        ckf_errors_phi[test].reserve(steps_per_test);
        ckf_errors_p[test].reserve(steps_per_test);

        // SRCF тест
        auto srcf_start = high_resolution_clock::now();

        for (int k = 0; k < steps_per_test; ++k) {
            double t = k * dt;

            auto A = model2::A(dt);
            auto B = model2::B(dt);
            auto C = model2::C();
            auto D = model2::D();
            auto Q = model2::Q(dt);
            auto R = model2::R();
            auto u = model2::u(t, ControlScenario::SINE_WAVE);

            // Истинное состояние с шумами
            x_true = A * x_true + B * u + model2::w(t, dt, true);
            Eigen::Vector2d y = C * x_true + model2::v(t, true);

            // SRCF шаг
            srcf_filter.step(A, B, C, D, Q, R, u, y);

            // Ошибки по компонентам
            double error_phi = std::abs(x_true(0) - srcf_filter.state()(0));
            double error_p = std::abs(x_true(1) - srcf_filter.state()(1));
            double total_error = std::sqrt(error_phi*error_phi + error_p*error_p);

            srcf_errors_phi[test].push_back(error_phi);
            srcf_errors_p[test].push_back(error_p);

            srcf_stats.total_error_phi += error_phi;
            srcf_stats.total_error_p += error_p;

            if (total_error > srcf_stats.max_error) srcf_stats.max_error = total_error;
        }

        auto srcf_end = high_resolution_clock::now();
        auto srcf_duration = duration_cast<microseconds>(srcf_end - srcf_start);

        // CKF тест
        auto ckf_start = high_resolution_clock::now();

        // Сброс состояния для нового теста
        x_true = Eigen::Vector2d::Random() * 3.0;
        x_ckf = x_true + Eigen::Vector2d::Random() * 1.0;
        ckf_filter.initialize(x_ckf, P0);

        for (int k = 0; k < steps_per_test; ++k) {
            double t = k * dt;

            auto A = model2::A(dt);
            auto B = model2::B(dt);
            auto C = model2::C();
            auto D = model2::D();
            auto Q = model2::Q(dt);
            auto R = model2::R();
            auto u = model2::u(t, ControlScenario::SINE_WAVE);

            // Истинное состояние с шумами
            x_true = A * x_true + B * u + model2::w(t, dt, true);
            Eigen::Vector2d y = C * x_true + model2::v(t, true);

            // CKF шаг
            ckf_filter.step(A, B, C, D, Q, R, u, y);

            // Ошибки по компонентам
            double error_phi = std::abs(x_true(0) - ckf_filter.state()(0));
            double error_p = std::abs(x_true(1) - ckf_filter.state()(1));
            double total_error = std::sqrt(error_phi*error_phi + error_p*error_p);

            ckf_errors_phi[test].push_back(error_phi);
            ckf_errors_p[test].push_back(error_p);

            ckf_stats.total_error_phi += error_phi;
            ckf_stats.total_error_p += error_p;

            if (total_error > ckf_stats.max_error) ckf_stats.max_error = total_error;

            // Проверка положительной определенности на последнем шаге
            if (k == steps_per_test - 1) {
                Eigen::Matrix2d P_final = ckf_filter.covariance();
                Eigen::LLT<Eigen::Matrix2d> llt_check(P_final);
                if (llt_check.info() != Eigen::Success) {
                    ckf_stats.positive_definite = false;
                }
                ckf_stats.condition_number = P_final.norm() * P_final.inverse().norm();
            }
        }

        auto ckf_end = high_resolution_clock::now();
        auto ckf_duration = duration_cast<microseconds>(ckf_end - ckf_start);

        // Обновление статистики времени
        srcf_stats.total_time += srcf_duration.count() / (double)steps_per_test;
        ckf_stats.total_time += ckf_duration.count() / (double)steps_per_test;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "SRCF: avg φ err=" << srcf_stats.total_error_phi / ((test+1)*steps_per_test)
                  << ", avg p err=" << srcf_stats.total_error_p / ((test+1)*steps_per_test)
                  << ", time/step=" << srcf_duration.count() / (double)steps_per_test << "μs\n";
        std::cout << "CKF:  avg φ err=" << ckf_stats.total_error_phi / ((test+1)*steps_per_test)
                  << ", avg p err=" << ckf_stats.total_error_p / ((test+1)*steps_per_test)
                  << ", time/step=" << ckf_duration.count() / (double)steps_per_test << "μs\n";
    }

    // Расчет итоговых статистик
    int total_steps = num_tests * steps_per_test;

    double srcf_avg_error_phi = srcf_stats.total_error_phi / total_steps;
    double srcf_avg_error_p = srcf_stats.total_error_p / total_steps;
    double srcf_avg_time = srcf_stats.total_time / num_tests;

    double ckf_avg_error_phi = ckf_stats.total_error_phi / total_steps;
    double ckf_avg_error_p = ckf_stats.total_error_p / total_steps;
    double ckf_avg_time = ckf_stats.total_time / num_tests;

    // Итоги
    std::cout << "\n=== FINAL RESULTS (" << num_tests << " tests, " << steps_per_test << " steps each) ===\n";
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nSRCF (Square Root Covariance Filter):\n";
    std::cout << "  Average error φ: " << srcf_avg_error_phi << " rad\n";
    std::cout << "  Average error p: " << srcf_avg_error_p << " rad/s\n";
    std::cout << "  ARMSE: " << std::sqrt((srcf_avg_error_phi*srcf_avg_error_phi +
                                           srcf_avg_error_p*srcf_avg_error_p)/2.0) << "\n";
    std::cout << "  Average time per step: " << srcf_avg_time << " μs\n";
    std::cout << "  Max frequency: " << 1e6 / srcf_avg_time << " Hz\n";
    std::cout << "  Max total error: " << srcf_stats.max_error << "\n";

    std::cout << "\nCKF (Conventional Kalman Filter):\n";
    std::cout << "  Average error φ: " << ckf_avg_error_phi << " rad\n";
    std::cout << "  Average error p: " << ckf_avg_error_p << " rad/s\n";
    std::cout << "  ARMSE: " << std::sqrt((ckf_avg_error_phi*ckf_avg_error_phi +
                                           ckf_avg_error_p*ckf_avg_error_p)/2.0) << "\n";
    std::cout << "  Average time per step: " << ckf_avg_time << " μs\n";
    std::cout << "  Max frequency: " << 1e6 / ckf_avg_time << " Hz\n";
    std::cout << "  Max total error: " << ckf_stats.max_error << "\n";
    std::cout << "  Covariance condition number: " << ckf_stats.condition_number << "\n";
    std::cout << "  Covariance positive definite: " << (ckf_stats.positive_definite ? "YES ✓" : "NO ✗") << "\n";

    std::cout << "\n=== COMPARISON ===\n";
    double error_ratio_phi = ckf_avg_error_phi / srcf_avg_error_phi;
    double error_ratio_p = ckf_avg_error_p / srcf_avg_error_p;
    double time_ratio = ckf_avg_time / srcf_avg_time;

    std::cout << "Error ratio φ (CKF/SRCF): " << error_ratio_phi;
    if (error_ratio_phi > 1.0) std::cout << " (SRCF is " << std::setprecision(2) << error_ratio_phi << "x better)";
    std::cout << "\n";

    std::cout << "Error ratio p (CKF/SRCF): " << error_ratio_p;
    if (error_ratio_p > 1.0) std::cout << " (SRCF is " << std::setprecision(2) << error_ratio_p << "x better)";
    std::cout << "\n";

    std::cout << "Speed ratio (CKF/SRCF): " << time_ratio;
    if (time_ratio > 1.0) std::cout << " (SRCF is " << std::setprecision(2) << time_ratio << "x faster)";
    else std::cout << " (CKF is " << std::setprecision(2) << 1.0/time_ratio << "x faster)";
    std::cout << "\n";

    // Согласно статье Verhaegen & Van Dooren (1986)
    std::cout << "\n=== Theoretical expectations (Verhaegen & Van Dooren, 1986) ===\n";
    std::cout << "1. SRCF should be more numerically stable (maintains P > 0)\n";
    std::cout << "2. SRCF handles ill-conditioned R_k^e better\n";
    std::cout << "3. CKF may diverge if A is unstable and symmetry is lost\n";
    std::cout << "4. SRCF requires more operations but is more reliable\n";

    // Визуализация трендов ошибок
    std::cout << "\n=== Error Trends ===\n";
    std::cout << "To visualize error convergence, consider plotting:\n";
    std::cout << "1. φ error over time for both filters\n";
    std::cout << "2. p error over time for both filters\n";
    std::cout << "3. Covariance matrix condition number over time\n";
}

void test_stability()
{
    using namespace model2;

    std::cout << "\n=== Stability Test ===\n";

    // Тест с плохой начальной оценкой
    Eigen::Vector2d x_true(0.0, 0.0);
    Eigen::Vector2d x_est(10.0, 5.0);  // Большая начальная ошибка!

    Eigen::Matrix2d P0;
    P0 << 100.0, 0.0,      // Большая начальная неопределенность
            0.0, 100.0;

    kalman::SRCF filter(x_est, P0);

    double dt = 0.01;
    const int steps = 500;

    for (int k = 0; k < steps; ++k)
    {
        double t = k * dt;

        auto A = model2::A(dt);
        auto B = model2::B(dt);
        auto C = model2::C();
        auto D = model2::D();
        auto Q = model2::Q(dt);
        auto R = model2::R();
        auto u = model2::u(t, ControlScenario::ZERO_HOLD);

        x_true = A * x_true + B * u;
        Eigen::Vector2d y = C * x_true;

        filter.step(A, B, C, D, Q, R, u, y);

        if (k % 50 == 0)
        {
            std::cout << "k=" << k
                      << " | True: [" << x_true.transpose() << "]"
                      << " | Est: [" << filter.state().transpose() << "]"
                      << " | Error: " << (x_true - filter.state()).norm()
                      << std::endl;
        }
    }
}

void test_SRCF_with_different_noise_safe()
{
    using namespace model2;

    std::cout << "\n=== Testing SRCF with different noise parameters (SAFE) ===\n";

    double dt = 0.01;
    const int steps = 1000;

    // Разные уровни шума измерений
    std::vector<double> R_scales = {0.1, 1.0, 10.0, 100.0};

    for (double R_scale : R_scales) {
        std::cout << "\n--- Testing with R_scale = " << R_scale << " ---\n";

        Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
        Eigen::Vector2d x_est = Eigen::Vector2d::Zero();

        Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

        kalman::SRCF filter(x_est, P0);

        double total_error = 0.0;
        double max_det = 0.0;
        Eigen::Matrix2d final_P;

        for (int k = 0; k < steps; ++k) {
            double t = k * dt;

            auto A = model2::A(dt);
            auto B = model2::B(dt);
            auto C = model2::C();
            auto D = model2::D();
            auto Q = model2::Q(dt);

            // Масштабируем R
            Eigen::Matrix2d R = model2::R() * R_scale;

            auto u = model2::u(t, ControlScenario::SINE_WAVE);

            // Истинное состояние (без шума)
            x_true = A * x_true + B * u;
            Eigen::Vector2d y = C * x_true;

            // Шаг фильтра
            filter.step(A, B, C, D, Q, R, u, y);

            double error = (x_true - filter.state()).norm();
            total_error += error;

            // Мониторинг ковариации ТОЛЬКО НА ПОСЛЕДНЕМ ШАГЕ
            if (k == steps - 1) {
                final_P = filter.covariance();  // КОПИРУЕМ ЗНАЧЕНИЕ, а не ссылку!
                double det = std::abs(final_P.determinant());
                max_det = det;
            }
        }

        std::cout << "Average error: " << total_error / steps << "\n";
        std::cout << "Final |det(P)|: " << max_det << "\n";
        std::cout << "Final P:\n" << final_P << "\n";

        if (final_P.norm() < 1e-10) {
            std::cout << "WARNING: Covariance is too small!\n";
        }
    }
}

void test_SRCF_with_different_noise()
{
    using namespace model2;

    std::cout << "\n=== Testing SRCF with different noise parameters ===\n";

    double dt = 0.01;
    const int steps = 1000;

    // Разные уровни шума измерений
    std::vector<double> R_scales = {0.1, 1.0, 10.0, 100.0};

    for (double R_scale : R_scales) {
        std::cout << "\n--- Testing with R_scale = " << R_scale << " ---\n";

        Eigen::Vector2d x_true = Eigen::Vector2d::Zero();
        Eigen::Vector2d x_est = Eigen::Vector2d::Zero();

        Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

        kalman::SRCF filter(x_est, P0);

        double total_error = 0.0;
        Eigen::Matrix2d max_P = Eigen::Matrix2d::Zero();
        double max_det = 0.0;

        for (int k = 0; k < steps; ++k) {
            double t = k * dt;

            auto A = model2::A(dt);
            auto B = model2::B(dt);
            auto C = model2::C();
            auto D = model2::D();
            auto Q = model2::Q(dt);

            // Масштабируем R
            Eigen::Matrix2d R = model2::R() * R_scale;

            auto u = model2::u(t, ControlScenario::SINE_WAVE);

            // Истинное состояние (без шума)
            x_true = A * x_true + B * u;
            Eigen::Vector2d y = C * x_true;

            // Шаг фильтра
            filter.step(A, B, C, D, Q, R, u, y);

            double error = (x_true - filter.state()).norm();
            total_error += error;

            // Мониторинг ковариации
            Eigen::Matrix2d P = filter.covariance();
            double det = std::abs(P.determinant());
            if (det > max_det) {
                max_det = det;
                max_P = P;
            }
        }

        std::cout << "Average error: " << total_error / steps << "\n";
        std::cout << "Max |det(P)|: " << max_det << "\n";
        std::cout << "Final P:\n" << filter.covariance() << "\n";
        std::cout << "Final P determinant: " << filter.covariance().determinant() << "\n";

        // Проверим, не слишком ли мала ковариация
        if (filter.covariance().norm() < 1e-10) {
            std::cout << "WARNING: Covariance is too small! Filter may be overconfident.\n";
        }
    }
}

void check_real_R()
{
    using namespace model2;

    std::cout << "\n=== Checking REAL R matrix ===\n";

    auto R_val = model2::R();
    std::cout << "Direct R() output:\n" << R_val << "\n";
    std::cout << "Type: " << typeid(R_val).name() << "\n";
    std::cout << "Size: " << R_val.rows() << "x" << R_val.cols() << "\n";

    // Проверим поэлементно
    std::cout << "R(0,0) = " << R_val(0,0) << " (expected: " << 0.01*0.01 << ")\n";
    std::cout << "R(1,1) = " << R_val(1,1) << " (expected: " << 0.02*0.02 << ")\n";

    // Проверим LLT
    Eigen::LLT<Eigen::Matrix2d> llt(R_val);
    std::cout << "LLT info: " << llt.info() << " (0=Success)\n";

    // Попробуем явно создать матрицу
    Eigen::Matrix2d R_manual;
    R_manual << 0.0001, 0.0,
            0.0, 0.0004;
    std::cout << "\nManual R:\n" << R_manual << "\n";

    Eigen::LLT<Eigen::Matrix2d> llt2(R_manual);
    std::cout << "Manual R LLT info: " << llt2.info() << "\n";
}

void debug_SRCF_one_iteration()
{
    using namespace model2;

    std::cout << "\n=== Debug SRCF One Iteration ===\n";

    double dt = 0.01;

    // 1. Создаем простой фильтр
    Eigen::Vector2d x_true(0.1, 0.2);
    Eigen::Vector2d x_est = x_true;
    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity() * 0.1;

    kalman::SRCF filter(x_est, P0);

    // 2. Получаем матрицы системы
    auto A = model2::A(dt);
    auto B = model2::B(dt);
    auto C = model2::C();
    auto D = model2::D();
    auto Q = model2::Q(dt);
    auto R_fixed = model2::R() * 0.1;  // Проблемный случай
    check_real_R();
//    Eigen::Matrix2d R_fixed;
//    R_fixed << 0.001, 0.0,
//            0.0, 0.001;

    auto u = model2::u(0.0, ControlScenario::ZERO_HOLD);

    // 3. Вычисляем истинное состояние и измерение
    x_true = A * x_true + B * u;
    Eigen::Vector2d y = C * x_true;

    std::cout << "Initial state: " << filter.state().transpose() << "\n";
    std::cout << "True state:    " << x_true.transpose() << "\n";
    std::cout << "Measurement y: " << y.transpose() << "\n";

    // 4. Сравним с CKF на одном шаге
    kalman::CKF ckf(filter.state(), P0);

    // SRCF шаг
    filter.step(A, B, C, D, Q, R_fixed, u, y);

    // CKF шаг
    ckf.step(A, B, C, D, Q, R_fixed, u, y);

    std::cout << "\nAfter one step:\n";
    std::cout << "True state:      " << x_true.transpose() << "\n";
    std::cout << "SRCF estimate:   " << filter.state().transpose() << "\n";
    std::cout << "CKF estimate:    " << ckf.state().transpose() << "\n";

    std::cout << "\nErrors:\n";
    std::cout << "SRCF error: " << (x_true - filter.state()).norm() << "\n";
    std::cout << "CKF error:  " << (x_true - ckf.state()).norm() << "\n";

    // 5. Проверим аналитически, что должно получиться
    // Для идеальных измерений фильтр должен дать оценку = истинное состояние
    std::cout << "\nExpected: filter should estimate exactly the true state\n";
    std::cout << "since measurements are perfect (no noise).\n";
}

// Тестовая версия SRCF::step с подробным выводом
void test_SRCF_step_verbose() {
    std::cout << "\n=== Verbose SRCF Step Test ===\n";

    // Те же параметры
    const int n = 2, p = 1, m = 1;
    double dt = 0.1;

    Eigen::Matrix2d A;
    A << 1.0, dt, 0.0, 1.0;
    Eigen::MatrixXd B(2, 1);
    B << 0.5*dt*dt, dt;
    Eigen::MatrixXd C(1, 2);
    C << 1.0, 0.0;
    Eigen::MatrixXd D(2, 1);
    D << 0.0, 0.0;
    Eigen::Matrix2d Q;
    Q << 0.01, 0.0, 0.0, 0.01;
    Eigen::MatrixXd R(1, 1);
    R << 0.1;

    Eigen::Vector2d x0(0.0, 0.0);
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0, 0.0, 0.5;

    // Ручная реализация одного шага SRCF с выводом
    Eigen::VectorXd x = x0;
    Eigen::MatrixXd S = P0.llt().matrixL();

    std::cout << "Initial S:\n" << S << "\n";

    // 1. Квадратные корни
    Eigen::MatrixXd SQ = Q.llt().matrixL();
    Eigen::MatrixXd SR = R.llt().matrixL();

    std::cout << "SQ:\n" << SQ << "\n";
    std::cout << "SR:\n" << SR << "\n";

    // 2. Prearray
    Eigen::MatrixXd prearray(p + n, p + n + m);
    std::cout << "prearray size: " << prearray.rows() << "x" << prearray.cols() << "\n";

    prearray.setZero();

    // Заполняем блоки
    std::cout << "Filling prearray blocks...\n";

    // Блок [0,0]: SR (p x p)
    prearray.block(0, 0, p, p) = SR;
    std::cout << "Block [0,0] SR: " << SR.rows() << "x" << SR.cols() << "\n";

    // Блок [0,p]: C*S (p x n)
    Eigen::MatrixXd C_times_S = C * S;
    std::cout << "C*S: " << C_times_S.rows() << "x" << C_times_S.cols() << "\n";
    prearray.block(0, p, p, n) = C_times_S;

    // Блок [p,p]: A*S (n x n)
    Eigen::MatrixXd A_times_S = A * S;
    std::cout << "A*S: " << A_times_S.rows() << "x" << A_times_S.cols() << "\n";
    prearray.block(p, p, n, n) = A_times_S;

    // Блок [p, p+n]: B*SQ (n x m)
    Eigen::MatrixXd B_times_SQ = B * SQ;
    std::cout << "B*SQ: " << B_times_SQ.rows() << "x" << B_times_SQ.cols() << "\n";
    prearray.block(p, p + n, n, m) = B_times_SQ;

    std::cout << "Prearray:\n" << prearray << "\n";

    // 3. QR разложение
    std::cout << "\nPerforming QR decomposition...\n";
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(prearray.transpose());
    Eigen::MatrixXd R_mat = qr.matrixQR().triangularView<Eigen::Upper>();

    std::cout << "R_mat size: " << R_mat.rows() << "x" << R_mat.cols() << "\n";

    // Берем первые (p+n) строк и транспонируем
    Eigen::MatrixXd postarray = R_mat.topRows(p + n).transpose();
    std::cout << "Postarray size: " << postarray.rows() << "x" << postarray.cols() << "\n";
    std::cout << "Postarray:\n" << postarray << "\n";

    // 4. Извлекаем блоки
    Eigen::MatrixXd S_Re = postarray.block(0, 0, p, p);
    Eigen::MatrixXd G = postarray.block(p, 0, n, p);
    Eigen::MatrixXd S_new = postarray.block(p, p, n, n);

    std::cout << "S_Re:\n" << S_Re << "\n";
    std::cout << "G:\n" << G << "\n";
    std::cout << "S_new:\n" << S_new << "\n";

    // 5. Обновление состояния
    Eigen::VectorXd u(1);
    u << 1.0;
    Eigen::VectorXd y(1);
    y << 0.1;  // тестовое измерение

    Eigen::VectorXd innov = y - C * x;
    std::cout << "Innovation: " << innov.transpose() << "\n";

    Eigen::VectorXd z = S_Re.triangularView<Eigen::Lower>().solve(innov);
    std::cout << "z: " << z.transpose() << "\n";

    Eigen::VectorXd x_new = A * x - G * z + D * u;
    std::cout << "New state: " << x_new.transpose() << "\n";

    std::cout << "\n=== Manual step completed ===\n";
}
