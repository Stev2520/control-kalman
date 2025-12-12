#ifndef MODELS_HPP
#define MODELS_HPP

#pragma once
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <cmath>

namespace kalman_noise
{
    class NoiseGenerator
    {
    private:
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;

    public:
        NoiseGenerator(int seed = std::random_device{}())
            : generator(seed), distribution(0.0, 1.0) {}

        double gaussian() { return distribution(generator); }

        Eigen::VectorXd gaussianVector(int size)
        {
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i) res(i) = distribution(generator);
            return res;
        }

        Eigen::VectorXd noiseWithCovariance(const Eigen::MatrixXd& covariance)
        {
            return covariance.llt().matrixL() * gaussianVector(covariance.rows());
        }

        void setSeed(int seed) { generator.seed(seed); }
    };

    static NoiseGenerator noise_gen;
} // namespace kalman_noise

// ============================================================================
// МОДЕЛЬ 0: Упрощенная модель рыскания самолета (yaw model)
// ============================================================================
namespace model0
{
    // Параметры системы
    const double b = 1.0;           // Коэффициент управления (-k₁/J)
    const double sigma_w = 1.0;     // Стандартное отклонение шума процесса (умеренный шум)
    const double sigma_v = 1.0;     // Стандартное отклонение шума измерений

    // Переменные для генерации коррелированного шума (если нужно)
    static double last_w = 0.0;
    static double alpha = 0.9;      // Коэффициент корреляции

    enum class ControlScenario {
        ZERO_HOLD,      // u = 0 (автопилот)
        STEP_MANEUVER,  // Ступенчатое управление
        SINE_WAVE,      // Синусоидальное управление
        PULSE           // Импульсное управление
    };

    // Инициализация шума
    void reset_noise() {
        last_w = 0.0;
    }

    // Генерация коррелированного шума процесса
    double generate_correlated_noise() {
        double xi = kalman_noise::noise_gen.gaussian();
        last_w = alpha * last_w + sigma_w * sqrt(1 - alpha*alpha) * xi;
        return last_w;
    }

    // ------------------------------------------------------------------------
    // МАТРИЦЫ МОДЕЛИ
    // ------------------------------------------------------------------------

    // Матрица перехода состояния (дискретная)
    Eigen::MatrixXd A(double dt)
    {
        Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(2, 2);
        phi(0, 0) = 1.0;
        phi(0, 1) = dt;
        phi(1, 0) = 0.0;
        phi(1, 1) = 1.0;
        return phi;
    }

    // Матрица управления (дискретная)
    Eigen::MatrixXd B(double dt)
    {
        Eigen::MatrixXd psi(2, 1);
        psi(0) = psi(1) = b * dt;
        psi(0) *= 0.5 * dt;
        return psi;
    }

    // Матрица измерений
    Eigen::MatrixXd C(double t = 0.0)
    {
        return Eigen::MatrixXd::Identity(2, 2);
    }

    // Матрица прямого воздействия (обычно 0 для фильтра Калмана)
    Eigen::MatrixXd G(double dt = 0.0) {
        return Eigen::MatrixXd::Zero(2, 1);
    }

    // Матрица воздействия шума процесса (дискретная)
    Eigen::VectorXd D(double dt)
    {
        Eigen::VectorXd gamma = Eigen::VectorXd::Zero(2);
        gamma(0) = gamma(1) = dt;
        gamma(0) *= 0.5 * dt;
        return gamma;
    }

    // ------------------------------------------------------------------------
    // КОВАРИАЦИОННЫЕ МАТРИЦЫ
    // ------------------------------------------------------------------------
    // Ковариация шума процесса (дискретная)
    Eigen::MatrixXd Q(double dt) {
        // Q_disc = G * σ_w² * Gᵀ
        Eigen::MatrixXd gamma = D(dt);
        Eigen::MatrixXd Q_disc = gamma * gamma.transpose() * sigma_w * sigma_w;

        // Гарантируем положительную определенность
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Q_disc);
        if (solver.eigenvalues().minCoeff() < 1e-12) {
            // Добавить стабилизацию
            Q_disc += 1e-8 * Eigen::MatrixXd::Identity(Q_disc.rows(), Q_disc.cols());
        }
        return Q_disc;
    }

    // Ковариация шума измерений (дискретная)
    Eigen::MatrixXd R(double t = 0.0) {
        Eigen::Matrix2d R_mat = Eigen::Matrix2d::Identity() * sigma_v * sigma_v;

        // Гарантируем положительную определенность
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R_mat);
        if (solver.eigenvalues().minCoeff() < 1e-12) {
            // Добавить стабилизацию
            R_mat += 1e-8 * Eigen::MatrixXd::Identity(R_mat.rows(), R_mat.cols());
        }
        return R_mat;
    }

    Eigen::MatrixXd Q_const(double t = 0.0)
    {
        static const Eigen::MatrixXd Q_const = []() {
            Eigen::MatrixXd q(2, 2);
            q << 0.1, 0.0,
                    0.0, 0.2;
            return q;
        }();
        return Q_const;
    }

    Eigen::MatrixXd R_const(double t = 0.0)
    {
        static const Eigen::MatrixXd R_const = []() {
            Eigen::MatrixXd r(2, 2);
            r << 1.0, 0.0,
                    0.0, 1.0;
            return r;
        }();
        return R_const;
    }

    // ------------------------------------------------------------------------
    // СИГНАЛЫ
    // ------------------------------------------------------------------------

    // Входное управление u(t) (отклонение руля)
    Eigen::VectorXd u(double t,
                      ControlScenario scenario = ControlScenario::SINE_WAVE)
    {
        Eigen::VectorXd u_vec = Eigen::VectorXd::Zero(1);

        switch (scenario) {
            case ControlScenario::ZERO_HOLD:
                u_vec << 0.0;
                break;
            case ControlScenario::STEP_MANEUVER:
                // Ступенька на 2 секунде
                if (t >= 2.0) {
                    u_vec << 0.1;
                } else {
                    u_vec << 0.0;
                }
                break;
            case ControlScenario::SINE_WAVE:
                // Синусоидальное управление
                u_vec << 0.05 * sin(0.5 * t);
                break;
            case ControlScenario::PULSE:
                // Импульс на 3 секунде
                u_vec << 0.2 * exp(-(t - 3.0) * (t - 3.0) / 0.5);
                break;
        }
        return u_vec;
    }

    // Шум процесса w_k
    Eigen::VectorXd w(double t, double dt, bool noise = false) {
        if (!noise) {
            return Eigen::VectorXd::Zero(1);  // Скалярный шум
        }
        double omega = generate_correlated_noise();
        Eigen::VectorXd w_vec(1);
        w_vec << omega;
        return w_vec;
    }

    // Шум измерений v_k
    Eigen::VectorXd v(double t,
                      bool noise = false)
    {
        if (!noise) {
            return Eigen::VectorXd::Zero(2);
        }
        return kalman_noise::noise_gen.noiseWithCovariance(R(t));
    }

    // ------------------------------------------------------------------------
    // ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    // ------------------------------------------------------------------------

    Eigen::Vector2d true_dynamics(const Eigen::Vector2d& x,
                                  double t,
                                  double dt,
                                  ControlScenario scenario,
                                  bool add_noise = true)
    {
        Eigen::Vector2d x_next = A(dt) * x + D(dt) * u(t, scenario);

        if (add_noise) {
            Eigen::VectorXd w_noise = w(t, dt, true);
            // Преобразуем скалярный шум в векторный через B(dt)
            x_next += B(dt) * w_noise(0);
        }

        return x_next;
    }
} // namespace model0

//// ============================================================================
//// МОДЕЛЬ 2: Модель крена самолета (roll model)
//// ============================================================================
namespace model2
{
    const double L_phi = 2.5; // 1/с² (восстанавливающий момент)
    const double L_p = 1.0; // 1/с (демпфирование)
    const double L_delta = 15.0; // 1/с² (эффективность элеронов)
    const double g = 9.80665; // м/с²
    const double T = 0.01; // шаг дискретизации (100 Гц)

    const double sigma_w = 0.02;  // рад/с² (интенсивность шума процесса)
    const double sigma_g = 0.01;  // рад/с (шум гироскопа)
    const double sigma_a = 0.02;  // рад (шум акселерометра)

    // Переменные для коррелированного шума (скрытые)
    static double last_omega = 0.0;
    static double alpha = 0.98;  // коэффициент корреляции

    enum class ControlScenario {
        ZERO_HOLD,      // u=0 (автопилот)
        STEP_MANEUVER,  // ступенчатое управление
        SINE_WAVE,      // синусоида
        PULSE           // импульс
    };

    // Инициализация шума (можно вызывать в начале симуляции)
    void reset_noise() {
        last_omega = 0.0;
    }

    // Генерация коррелированного шума процесса
    double generate_correlated_noise() {
        // Модель AR(1): ω_k = α·ω_{k-1} + σ·√(1-α²)·ξ_k
        double xi = kalman_noise::noise_gen.gaussian();
        last_omega = alpha * last_omega + sigma_w * sqrt(1 - alpha * alpha) * xi;
        return last_omega;
    }

    // ------------------------------------------------------------------------
    // ОСНОВНЫЕ ФУНКЦИИ ДИСКРЕТИЗАЦИИ
    // ------------------------------------------------------------------------

    // Формулы для Φ(T), ψ(T), Γ(T)
    // Матрица перехода состояния
    Eigen::MatrixXd Phi(double dt) {
        Eigen::Matrix2d phi;
        double exp_term = exp(-0.5*dt);
        double sin_term = sin(1.5*dt);
        double cos_term = cos(1.5*dt);
        phi(0,0) = exp_term * (sin_term/3.0 + cos_term);
        phi(0,1) = exp_term * (2.0 * sin_term/3.0);
        phi(1,0) = exp_term * (-5.0 * sin_term/3.0);
        phi(1,1) = exp_term * (-sin_term/3.0 + cos_term);
        return phi;
    }

    // Входная матрица для управления (дискретная)
    Eigen::Vector2d Psi(double dt) {
        Eigen::Vector2d psi;
        double exp_term = exp(-0.5 * dt);
        double sin_term = sin(1.5 * dt);
        double cos_term = cos(1.5 * dt);
        psi(0) = -2 * sin_term * exp_term - 6 * cos_term * exp_term + 6;
        psi(1) = 10 * sin_term * exp_term;
        return psi;
    }

    // Матрица для шума процесса (дискретная)
    Eigen::Vector2d Gamma(double dt) {
        Eigen::Vector2d gamma;
        double exp_term = exp(-0.5 * dt);
        double sin_term = sin(1.5 * dt);
        double cos_term = cos(1.5 * dt);
        gamma(0) = -2.0/15.0 * sin_term * exp_term - 2.0/5.0 * cos_term * exp_term + 2.0/5.0;
        gamma(1) = 2.0/3.0 * sin_term * exp_term;
        return gamma;
    }

    // ------------------------------------------------------------------------
    // НЕПРЕРЫВНЫЕ МАТРИЦЫ (для справки)
    // ------------------------------------------------------------------------

    // Непрерывная матрица A для линеаризованной системы
    Eigen::Matrix2d A_continuous() {
        Eigen::Matrix2d A = Eigen::Matrix2d::Zero(2, 2);
        A << 0, 1,
            -L_phi, -L_p;
        return A;
    }

    // Непрерывная матрица B для управления для линеаризованной системы
    Eigen::Vector2d D_continuous() {
        Eigen::Vector2d D = Eigen::Vector2d::Zero(2);
        D << 0, L_delta;
        return D;
    }

    // Непрерывная матрица G для шума процесса для линеаризованной системы
    Eigen::Vector2d B_continuous() {
        Eigen::Vector2d B = Eigen::Vector2d::Zero(2);
        B << 0, 1;
        return B;
    }

    // ------------------------------------------------------------------------
    // МАТРИЦЫ ДЛЯ ФИЛЬТРА КАЛМАНА
    // ------------------------------------------------------------------------

    // Матрица перехода состояния (дискретная)
    Eigen::MatrixXd A(double dt)
    {
        return Phi(dt);
    }

    // Матрица управления (дискретная)
    Eigen::MatrixXd D(double dt) {
        Eigen::Vector2d psi = Psi(dt);
        Eigen::MatrixXd B_mat(2, 1);
        B_mat = psi;
        return B_mat;
    }

    // Матрица измерений C
    Eigen::Matrix2d C(double t = 0.0) {
        // Матрица измерений: y1 = p (угловая скорость), y2 = g*sin(φ) ≈ g*φ
        Eigen::MatrixXd C_mat = Eigen::MatrixXd::Zero(2, 2);
        C_mat << 0, 1,    // измеряем угловую скорость (вторая компонента состояния) (рад/с)
                g, 0;     // измеряем g*φ ≈ ускорение (м/с²) (первая компонента состояния)
        return C_mat;
    }

    // Матрица прямого воздействия
    Eigen::MatrixXd B(double dt = 0.0) {
        return Gamma(dt);
    }

    // ------------------------------------------------------------------------
    // КОВАРИАЦИОННЫЕ МАТРИЦЫ
    // ------------------------------------------------------------------------

    // Ковариация шума процесса (дискретная)
    Eigen::MatrixXd Q(double dt) {
        Eigen::MatrixXd Q_mat(1, 1);
        Q_mat << sigma_w * sigma_w * dt;
        return Q_mat;
    }

    // Ковариация шума измерений
    Eigen::MatrixXd R(double t = 0.0) {
        Eigen::Matrix2d R_mat;
        double rg = sigma_g * sigma_g;  // дисперсия гироскопа
        double ra = sigma_a * sigma_a;  // дисперсия акселерометра
        R_mat << rg, 0.0,
                0.0, ra;

        // Гарантируем положительную определенность
        if (R_mat.determinant() < 1e-12) {
            std::cout << "WARNING: R is near-singular, adding stabilization\n";
            R_mat += Eigen::Matrix2d::Identity() * 1e-6;
        }

        return R_mat;
    }

    // Ковариация шума процесса Q
    // Это мощность непрерывного белого шума w(t)
    Eigen::Matrix2d Q_const(double t = 0.0) {
        Eigen::Matrix2d Q_const;
        Q_const << 0.01, 0.0,
                0.0, 0.01;
        return Q_const;
    }

    // Ковариация шума измерений R
    Eigen::Matrix2d R_const(double t = 0.0) {
        Eigen::Matrix2d R_const;
        R_const << 0.1, 0.0,    // шум гироскопа (рад/с)
                0.0, 0.5;    // шум акселерометра (м/с²)
// Для более агрессивного фильтра (меньшая доверительность к измерениям)
//        R << 0.01, 0.0,    // Более низкий шум гироскопа
//                0.0, 0.05;    // Более низкий шум акселерометра
// Для более консервативного фильтра (большая доверительность к измерениям)
//        R << 1.0, 0.0,     // Высокий шум гироскопа
//                0.0, 2.0;     // Высокий шум акселерометра
        return R_const;
    }

    // ------------------------------------------------------------------------
    // СИГНАЛЫ
    // ------------------------------------------------------------------------

    // Входное управление u(t) (отклонение элеронов)
    Eigen::VectorXd u(double t, ControlScenario scenario = ControlScenario::SINE_WAVE)
    {
        Eigen::VectorXd u_vec(1);

        switch (scenario) {
            case ControlScenario::ZERO_HOLD:
                u_vec << 0.0;  // Автопилот
                break;

            case ControlScenario::STEP_MANEUVER:
                // Ступенька: на 2 секунде отклонение 0.1 рад
                if (t >= 2.0) {
                    u_vec << 0.1;
                }
                else {
                    u_vec << 0.0;
                }
                break;

            case ControlScenario::SINE_WAVE:
                // Синусоидальное управление
                u_vec << 0.05 * sin(0.5 * t);
                break;

            case ControlScenario::PULSE:
                // Импульс на 3 секунде
                u_vec << 0.2 * exp(-(t - 3.0) * (t - 3.0) / 0.5);
                break;
        }

        return u_vec;
    }

    // Шум процесса w_k
    Eigen::VectorXd w(double t, double dt, bool noise = false) {
        if (!noise) {
            return Eigen::VectorXd::Zero(2);
        }
        // w(0) = 0, w(1) = коррелированный шум
        Eigen::Vector2d w_vec;
        w_vec << 0.0, generate_correlated_noise();
        return w_vec;
    }

    // Шум измерений v_k
    Eigen::VectorXd v(double t,
                      bool noise = false) {
        if (!noise) {
            return Eigen::Vector2d::Zero();
        }
        return kalman_noise::noise_gen.noiseWithCovariance(R(t));
    }

    // ------------------------------------------------------------------------
    // ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    // ------------------------------------------------------------------------

    // Истинная динамика системы
    Eigen::Vector2d true_dynamics(const Eigen::Vector2d& x,
                                  double t,
                                  double dt,
                                  ControlScenario scenario,
                                  bool add_noise = true)
    {
        Eigen::Vector2d x_next = A(dt) * x + D(dt) * u(t, scenario);
        if (add_noise) {
            Eigen::VectorXd w_noise = w(t, dt, true);
            x_next += B(dt) * w_noise(0);
        }
        return x_next;
    }

    // Измерения
    Eigen::Vector2d measurement(const Eigen::Vector2d& x,
                                double t,
                                bool add_noise = true)
    {
        Eigen::Vector2d y = C(t) * x;

        if (add_noise) {
            y += v(t, true);
        }

        return y;
    }
} // namespace model2

#endif // MODELS_HPP