#include <iostream>
#include <Eigen/Dense>
#include "kalman.hpp"

void test_simple_SRCF() {
    std::cout << "=== Simple SRCF Test ===\n";

    const int n = 2;  // размер состояния
    const int p = 1;  // размер измерения
    const int m = 1;  // размер шума процесса

    double dt = 0.1;

    // Матрицы модели
    Eigen::Matrix2d A;
    A << 1.0, dt, 0.0, 1.0;

    Eigen::MatrixXd B(2, 1);
    B << 0.5*dt*dt, dt;

    Eigen::MatrixXd C(1, 2);
    C << 1.0, 0.0;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 1);

    // Ковариации - ВАЖНО: Q размером m×m = 1×1
    Eigen::MatrixXd Q(1, 1);
    Q << 0.01;  // скалярная ковариация шума процесса

    Eigen::MatrixXd R(1, 1);
    R << 0.1;

    // Начальные условия
    Eigen::Vector2d x0(0.0, 0.0);
    Eigen::Matrix2d P0;
    P0 << 1.0, 0.0, 0.0, 1.0;

    // Инициализация фильтра
    kalman::SRCF srcf(x0, P0);

    // Тестовый сигнал
    Eigen::VectorXd u(1);
    u << 1.0;

    std::cout << "Initial state: " << srcf.state().transpose() << std::endl;
    srand(42);

    for (int k = 0; k < 10; ++k) {
        static Eigen::Vector2d x_true = x0;
        x_true = A * x_true + B * u;

        Eigen::VectorXd measurement_noise(1);
        measurement_noise << 0.1 * (rand() / (double)RAND_MAX - 0.5);

        Eigen::VectorXd y(1);
        y = C * x_true + measurement_noise;

        // Шаг фильтра
        srcf.step(A, B, C, D, Q, R, u, y);

        std::cout << "\nStep " << k << ":\n";
        std::cout << "  True: " << x_true.transpose() << "\n";
        std::cout << "  Est:  " << srcf.state().transpose() << "\n";
        std::cout << "  Error: " << (x_true - srcf.state()).norm() << "\n";
    }
}