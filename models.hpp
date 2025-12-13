/**
 * @file models.hpp
 * @brief Модели систем для фильтра Калмана (самолетное применение)
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * @note Данный файл содержит модели двух систем:
 *       1. MODEL0 - Упрощенная модель рыскания самолета (yaw model)
 *       2. MODEL2 - Модель крена самолета (roll model)
 *       Каждая модель включает матрицы состояния, ковариации шумов,
 *       функции управления и генерации данных для симуляции.
 */

#ifndef MODELS_HPP
#define MODELS_HPP

#pragma once

#include <Eigen/Dense>
#include <random>
#include <functional>
#include <cmath>

// ============================================================================
// ГЕНЕРАТОР ШУМА
// ============================================================================

namespace model0 {
    void reset_noise_with_seed(int seed);
}

namespace model2 {
    void reset_noise_with_seed(int seed);
}

/**
 * @namespace kalman_noise
 * @brief Пространство имен для генерации шумов в фильтре Калмана
 */
namespace kalman_noise
{
    /**
     * @class NoiseGenerator
     * @brief Генератор гауссовского шума для фильтра Калмана
     *
     * @note Поддерживает генерацию скалярного шума, векторного шума
     *       и шума с заданной ковариационной матрицей.
     */
    class NoiseGenerator
    {
    private:
        std::default_random_engine generator; /**< Генератор случайных чисел */
        std::normal_distribution<double> distribution; /**< Нормальное распределение N(0,1) */

    public:
        /**
         * @brief Конструктор генератора шума
         * @param seed Начальное значение для генератора случайных чисел
         */
        explicit NoiseGenerator(int seed = std::random_device{}())
                : generator(seed), distribution(0.0, 1.0) {}

        /**
        * @brief Генерация скалярного гауссовского шума
        * @return double Значение шума из распределения N(0,1)
        */
        double gaussian() { return distribution(generator); }

        /**
         * @brief Генерация векторного гауссовского шума
         * @param size Размерность вектора
         * @return Eigen::VectorXd Вектор независимых гауссовских шумов N(0,1)
         */
        Eigen::VectorXd gaussianVector(int size)
        {
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i) res(i) = distribution(generator);
            return res;
        }

        /**
         * @brief Генерация шума с заданной ковариационной матрицей
         * @param covariance Ковариационная матрица шума
         * @return Eigen::VectorXd Вектор шума с заданной ковариацией
         *
         * @note Используется разложение Холецкого: v = L * ξ, где L = chol(C)
         */
        Eigen::VectorXd noiseWithCovariance(const Eigen::MatrixXd& covariance)
        {
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("NoiseGenerator: covariance matrix is not positive definite");
            }
            return llt.matrixL() * gaussianVector(covariance.rows());
        }

        /**
         * @brief Установка нового зерна для генератора
         * @param seed Новое значение зерна
         */
        void setSeed(int seed) { generator.seed(seed); }

        /**
         * @brief Получение текущего состояния генератора
         * @return std::default_random_engine& Ссылка на генератор
         */
        std::default_random_engine& getGenerator() {
            return generator;
        }

        void reset(int seed = std::random_device{}()) {
            generator.seed(seed);
            distribution.reset();
        }

        void reset_with_state(int seed = std::random_device{}()) {
            // Полный сброс: создаем новый объект
            *this = NoiseGenerator(seed);
        }

    };

    /**
     * @brief Статический экземпляр генератора шума для общего использования
     */
    static NoiseGenerator noise_gen;

    inline void reset_noise_generators(int seed = 42) {
        noise_gen.reset_with_state(seed);

        // Передаем seed в функции сброса
        model0::reset_noise_with_seed(seed);
        model2::reset_noise_with_seed(seed);
    }
} // namespace kalman_noise

// ============================================================================
// МОДЕЛЬ 0: Упрощенная модель рыскания самолета (yaw model)
// ============================================================================

/**
 * @namespace model0
 * @brief Упрощенная модель рыскания самолета
 *
 * @note Модель описывает динамику рыскания самолета:
 *       x = [ψ, r]ᵀ, где ψ - угол рыскания, r - скорость рыскания
 *       Уравнение: J·ṙ = -k₁·r + k₂·δ_r + w(t)
 *       где δ_r - отклонение руля направления
 */
namespace model0
{
    // Параметры системы
    const double b = 1.0;           /**< Коэффициент управления (-k₁/J) */
    const double sigma_w = 1.0;     /**< Стандартное отклонение шума процесса (умеренный шум) */
    const double sigma_v = 1.0;     /**< Стандартное отклонение шума измерений */

    // Переменные для генерации коррелированного шума
//    static double last_w = 0.0;     /**< Последнее значение коррелированного шума */
//    static double alpha = 0.9;      /**< Коэффициент корреляции AR(1) процесса */

    /**
     * @enum ControlScenario
     * @brief Сценарии управления для модели 0
     */
    enum class ControlScenario {
        ZERO_HOLD,      /**< u = 0 (автопилот, стабилизация) */
        STEP_MANEUVER,  /**< Ступенчатое управление */
        SINE_WAVE,      /**< Синусоидальное управление */
        PULSE           /**< Импульсное управление */
    };

    // Структура для хранения состояния генератора
    struct NoiseState {
        double last_w = 0.0;
        double alpha = 0.98;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;

        NoiseState(int seed = 42)
                : generator(seed), distribution(0.0, 1.0) {}

        void reset(int seed) {
            last_w = 0.0;
            generator.seed(seed);
            distribution.reset();
        }
    };

    // Thread-local состояние
    NoiseState& get_noise_state() {
        static thread_local NoiseState noise_state(42);  // Только для инициализации по умолчанию
        return noise_state;
    }

    void reset_noise_with_seed(int seed) {
        get_noise_state().reset(seed);
    }

    double generate_correlated_noise() {
        auto& state = get_noise_state();
        double xi = state.distribution(state.generator);
        state.last_w = state.alpha * state.last_w +
                       sigma_w * std::sqrt(1.0 - state.alpha * state.alpha) * xi;
        return state.last_w;
    }

    /**
     * @brief Сброс состояния генератора шума
     */
//    void reset_noise_with_seed(int seed) {
//        last_w = 0.0;
//        generator.seed(seed);
//    }

    /**
     * @brief Генерация коррелированного шума процесса по модели AR(1)
     * @return double Значение коррелированного шума
     *
     * @note Модель: ω_k = α·ω_{k-1} + σ·√(1-α²)·ξ_k
     */
//    double generate_correlated_noise()
//    {
//        double xi = distribution(generator);
//        last_w = alpha * last_w + sigma_w * std::sqrt(1.0 - alpha * alpha) * xi;
//        return last_w;
//    }

    // ------------------------------------------------------------------------
    // МАТРИЦЫ МОДЕЛИ
    // ------------------------------------------------------------------------

    /**
     * @brief Матрица перехода состояния (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица перехода размером 2×2
     *
     * @note Для модели двойного интегратора: x_{k+1} = [1, dt; 0, 1]·x_k
     */
    Eigen::MatrixXd A(double dt)
    {
        Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(2, 2);
        phi(0, 0) = 1.0;
        phi(0, 1) = dt;
        phi(1, 0) = 0.0;
        phi(1, 1) = 1.0;
        return phi;
    }

    /**
     * @brief Матрица управления (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица управления размером 2×1
     *
     * @note Для двойного интегратора с управлением: ψ = [0.5·dt², dt]ᵀ·b
     */
    Eigen::MatrixXd B(double dt)
    {
        Eigen::MatrixXd psi(2, 1);
        psi(0) = psi(1) = b * dt;
        psi(0) *= 0.5 * dt;
        return psi;
    }

    /**
     * @brief Матрица измерений
     * @param t Время (не используется, для совместимости)
     * @return Eigen::MatrixXd Единичная матрица измерений 2×2
     */
    Eigen::MatrixXd C(double t = 0.0)
    {
        return Eigen::MatrixXd::Identity(2, 2);
    }

    /**
     * @brief Матрица прямого воздействия (обычно нулевая для фильтра Калмана)
     * @param dt Шаг дискретизации (не используется)
     * @return Eigen::MatrixXd Нулевая матрица 2×1
     */
    Eigen::MatrixXd G(double dt = 0.0)
    {
        return Eigen::MatrixXd::Zero(2, 1);
    }

    /**
     * @brief Матрица воздействия шума процесса (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::VectorXd Вектор воздействия шума размером 2
     *
     * @note γ = [0.5·dt², dt]ᵀ для двойного интегратора
     */
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
    /**
     * @brief Ковариация шума процесса (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица ковариации 2×2
     *
     * @note Q_disc = γ·σ_w²·γᵀ с регуляризацией для положительной определенности
     */
    Eigen::MatrixXd Q(double dt)
    {
        Eigen::MatrixXd gamma = D(dt);
        Eigen::MatrixXd Q_disc = gamma * gamma.transpose() * sigma_w * sigma_w;

        // Гарантируем положительную определенность
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Q_disc);
        if (solver.eigenvalues().minCoeff() < 1e-12) {
            Q_disc += 1e-8 * Eigen::MatrixXd::Identity(Q_disc.rows(), Q_disc.cols());
        }
        return Q_disc;
    }

    /**
     * @brief Ковариация шума измерений (дискретная)
     * @param t Время (не используется, для совместимости)
     * @return Eigen::MatrixXd Матрица ковариации 2×2
     *
     * @note R = σ_v²·I с регуляризацией
     */
    Eigen::MatrixXd R(double t = 0.0)
    {
        Eigen::Matrix2d R_mat = Eigen::Matrix2d::Identity() * sigma_v * sigma_v;

        // Гарантируем положительную определенность
        if (R_mat.determinant() < 1e-12) {
            R_mat += 1e-8 * Eigen::Matrix2d::Identity();
        }
        return R_mat;
    }

    /**
     * @brief Константная ковариация шума процесса (для тестирования)
     * @param t Время (не используется)
     * @return Eigen::MatrixXd Фиксированная матрица ковариации 2×2
     */
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

    /**
     * @brief Константная ковариация шума измерений (для тестирования)
     * @param t Время (не используется)
     * @return Eigen::MatrixXd Фиксированная матрица ковариации 2×2
     */
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
    // СИГНАЛЫ УПРАВЛЕНИЯ И ШУМОВ
    // ------------------------------------------------------------------------

    /**
     * @brief Входное управление (отклонение руля направления)
     * @param t Текущее время (секунды)
     * @param scenario Сценарий управления
     * @return Eigen::VectorXd Вектор управления размером 1
     */
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

    /**
     * @brief Шум процесса
     * @param t Время (не используется)
     * @param dt Шаг времени (не используется)
     * @param noise Флаг добавления шума
     * @return Eigen::VectorXd Вектор шума процесса размером 1
     */
    Eigen::VectorXd w(double t, double dt, bool noise = false)
    {
        if (!noise) {
            return Eigen::VectorXd::Zero(1);  // Скалярный шум
        }
        double omega = generate_correlated_noise();
        Eigen::VectorXd w_vec(1);
        w_vec << omega;
        return w_vec;
    }

    /**
    * @brief Шум измерений
    * @param t Время (для вычисления ковариации)
    * @param noise Флаг добавления шума
    * @return Eigen::VectorXd Вектор шума измерений размером 2
    */
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

    /**
     * @brief Истинная динамика системы
     * @param x Текущее состояние [ψ, r]ᵀ
     * @param t Текущее время (секунды)
     * @param dt Шаг времени (секунды)
     * @param scenario Сценарий управления
     * @param add_noise Флаг добавления шума процесса
     * @return Eigen::Vector2d Следующее состояние системы
     */
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
//    Eigen::Vector2d true_dynamics(const Eigen::Vector2d& x,
//                                  double t,
//                                  double dt,
//                                  ControlScenario scenario,
//                                  bool add_noise = true)
//    {
//        Eigen::Vector2d x_next = A(dt) * x + B(dt) * u(t, scenario)(0);
//
//        if (add_noise) {
//            Eigen::VectorXd w_noise = w(t, dt, true);
//            // Для двойного интегратора: шум прилагается как управление
//            x_next += D(dt) * w_noise(0);
//        }
//
//        return x_next;
//    }
} // namespace model0

//// ============================================================================
//// МОДЕЛЬ 2: Модель крена самолета (roll model)
//// ============================================================================

/**
 * @namespace model2
 * @brief Модель крена самолета
 *
 * @note Модель описывает динамику крена самолета:
 *       x = [φ, p]ᵀ, где φ - угол крена, p - скорость крена
 *       Уравнения: φ̇ = p
 *                  ṗ = -L_φ·φ - L_p·p + L_δ·δ_a + w(t)
 *       где δ_a - отклонение элеронов
 */
namespace model2
{
    // Параметры системы
    const double L_phi = 2.5;       /**< 1/с² (восстанавливающий момент) */
    const double L_p = 1.0;         /**< 1/с (демпфирование) */
    const double L_delta = 15.0;    /**< 1/с² (эффективность элеронов) */
    const double g = 9.80665;       /**< м/с² (ускорение свободного падения) */
    const double sigma_w = 0.02;    /**< рад/с² (интенсивность шума процесса) */
    const double sigma_g = 0.01;    /**< рад/с (шум гироскопа) */
    const double sigma_a = 0.02;    /**< рад (шум акселерометра) */

    // Переменные для коррелированного шума
//    static double last_omega = 0.0; /**< Последнее значение коррелированного шума */
//    static bool is_initialized = false;
//    static double alpha = 0.98;     /**< Коэффициент корреляции AR(1) процесса */

    /**
     * @enum ControlScenario
     * @brief Сценарии управления для модели 2
     */
    enum class ControlScenario {
        ZERO_HOLD,      /**< u=0 (автопилот, стабилизация) */
        STEP_MANEUVER,  /**< Ступенчатое управление */
        SINE_WAVE,      /**< Синусоидальное управление */
        PULSE           /**< Импульсное управление */
    };

    // Структура для хранения состояния генератора
    struct NoiseState {
        double last_w = 0.0;
        double alpha = 0.98;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;

        NoiseState(int seed = 42)
                : generator(seed), distribution(0.0, 1.0) {}

        void reset(int seed) {
            last_w = 0.0;
            generator.seed(seed);
            distribution.reset();
        }
    };

    // Thread-local состояние
    NoiseState& get_noise_state() {
        static thread_local NoiseState noise_state(42);  // Только для инициализации по умолчанию
        return noise_state;
    }

    void reset_noise_with_seed(int seed) {
        get_noise_state().reset(seed);
    }

    double generate_correlated_noise() {
        auto& state = get_noise_state();
        double xi = state.distribution(state.generator);
        state.last_w = state.alpha * state.last_w +
                       sigma_w * std::sqrt(1.0 - state.alpha * state.alpha) * xi;
        return state.last_w;
    }

    /**
     * @brief Генерация коррелированного шума процесса по модели AR(1)
     * @return double Значение коррелированного шума
     *
     * @note Модель: ω_k = α·ω_{k-1} + σ·√(1-α²)·ξ_k
     */
//    double generate_correlated_noise()
//    {
//        double xi = distribution(generator);
//        last_w = alpha * last_w + sigma_w * std::sqrt(1.0 - alpha * alpha) * xi;
//        return last_w;
//    }

    // ------------------------------------------------------------------------
    // ОСНОВНЫЕ ФУНКЦИИ ДИСКРЕТИЗАЦИИ
    // ------------------------------------------------------------------------

    // Формулы для Φ(T), ψ(T), Γ(T)
    /**
     * @brief Матрица перехода состояния (аналитическое решение)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::Matrix2d Матрица перехода 2×2
     *
     * @note Аналитическое решение для системы: ẋ = A·x, где A = [0, 1; -L_φ, -L_p]
     *       Собственные значения: λ = -0.5 ± 1.5i
     */
    Eigen::MatrixXd Phi(double dt)
    {
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

    /**
     * @brief Вектор влияния управления (дискретный)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::Vector2d Вектор управления 2×1
     */
    Eigen::Vector2d Psi(double dt)
    {
        Eigen::Vector2d psi;
        double exp_term = exp(-0.5 * dt);
        double sin_term = sin(1.5 * dt);
        double cos_term = cos(1.5 * dt);
        psi(0) = -2 * sin_term * exp_term - 6 * cos_term * exp_term + 6;
        psi(1) = 10 * sin_term * exp_term;
        return psi;
    }

    /**
     * @brief Матрица воздействия шума процесса (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::Vector2d Вектор воздействия шума 2×1
     */
    Eigen::Vector2d Gamma(double dt)
    {
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

    /**
     * @brief Непрерывная матрица состояния A
     * @return Eigen::Matrix2d Матрица A непрерывной системы
     */
    Eigen::Matrix2d A_continuous()
    {
        Eigen::Matrix2d A = Eigen::Matrix2d::Zero(2, 2);
        A << 0, 1,
                -L_phi, -L_p;
        return A;
    }

    /**
    * @brief Непрерывная матрица управления D
    * @return Eigen::Vector2d Вектор управления непрерывной системы
    */
    Eigen::Vector2d D_continuous()
    {
        Eigen::Vector2d D = Eigen::Vector2d::Zero(2);
        D << 0, L_delta;
        return D;
    }

    /**
     * @brief Непрерывная матрица воздействия шума B
     * @return Eigen::Vector2d Вектор воздействия шума непрерывной системы
     */
    Eigen::Vector2d B_continuous()
    {
        Eigen::Vector2d B = Eigen::Vector2d::Zero(2);
        B << 0, 1;
        return B;
    }

    // ------------------------------------------------------------------------
    // МАТРИЦЫ ДЛЯ ФИЛЬТРА КАЛМАНА
    // ------------------------------------------------------------------------

    /**
     * @brief Матрица перехода состояния (дискретная) для фильтра Калмана
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица перехода 2×2
     */
    Eigen::MatrixXd A(double dt)
    {
        return Phi(dt);
    }

    /**
     * @brief Матрица управления (дискретная) для фильтра Калмана
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица управления 2×1
     */
    Eigen::MatrixXd D(double dt) {
        Eigen::Vector2d psi = Psi(dt);
        Eigen::MatrixXd B_mat(2, 1);
        B_mat = psi;
        return B_mat;
    }

    /**
     * @brief Матрица измерений C
     * @param t Время (не используется, для совместимости)
     * @return Eigen::Matrix2d Матрица измерений 2×2
     *
     * @note y₁ = p (угловая скорость крена, гироскоп)
     *       y₂ = g·sin(φ) ≈ g·φ (ускорение, акселерометр)
     */
    Eigen::Matrix2d C(double t = 0.0) {
        // Матрица измерений: y1 = p (угловая скорость), y2 = g*sin(φ) ≈ g*φ
        Eigen::MatrixXd C_mat = Eigen::MatrixXd::Zero(2, 2);
        C_mat << 0, 1,    // измеряем угловую скорость (вторая компонента состояния) (рад/с)
                g, 0;     // измеряем g*φ ≈ ускорение (м/с²) (первая компонента состояния)
        return C_mat;
    }

    /**
     * @brief Матрица воздействия шума процесса для фильтра Калмана
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица воздействия шума 2×1
     */
    Eigen::MatrixXd B(double dt = 0.0) {
        return Gamma(dt);
    }

    // ------------------------------------------------------------------------
    // КОВАРИАЦИОННЫЕ МАТРИЦЫ
    // ------------------------------------------------------------------------

    /**
     * @brief Ковариация шума процесса (дискретная)
     * @param dt Шаг дискретизации (секунды)
     * @return Eigen::MatrixXd Матрица ковариации 1×1
     *
     * @note Для скалярного шума: Q = σ_w²·dt
     */
    Eigen::MatrixXd Q(double dt)
    {
        Eigen::MatrixXd Q_mat(1, 1);
        Q_mat << sigma_w * sigma_w * dt;
        return Q_mat;
    }

    /**
     * @brief Ковариация шума измерений
     * @param t Время (не используется, для совместимости)
     * @return Eigen::Matrix2d Матрица ковариации 2×2
     *
     * @note R = diag(σ_g², σ_a²) с регуляризацией
     */
    Eigen::MatrixXd R(double t = 0.0)
    {
        Eigen::Matrix2d R_mat;
        double rg = sigma_g * sigma_g;  // дисперсия гироскопа
        double ra = sigma_a * sigma_a;  // дисперсия акселерометра
        R_mat << rg, 0.0,
                0.0, ra;

        // Гарантируем положительную определенность
        if (R_mat.determinant() < 1e-12) {
            std::cout << "[MODEL2] WARNING: R is near-singular, adding stabilization\n";
            R_mat += Eigen::Matrix2d::Identity() * 1e-6;
        }

        return R_mat;
    }

    /**
     * @brief Константная ковариация шума процесса (для тестирования)
     * @param t Время (не используется)
     * @return Eigen::Matrix2d Фиксированная матрица ковариации 2×2
     */
    Eigen::Matrix2d Q_const(double t = 0.0)
    {
        Eigen::Matrix2d Q_const;
        Q_const << 0.01, 0.0,
                0.0, 0.01;
        return Q_const;
    }

    /**
     * @brief Константная ковариация шума измерений (для тестирования)
     * @param t Время (не используется)
     * @return Eigen::Matrix2d Фиксированная матрица ковариации 2×2
     */
    Eigen::Matrix2d R_const(double t = 0.0)
    {
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

    /**
    * @brief Входное управление (отклонение элеронов)
    * @param t Текущее время (секунды)
    * @param scenario Сценарий управления
    * @return Eigen::VectorXd Вектор управления размером 1
    */
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

    /**
     * @brief Шум процесса
     * @param t Время (не используется)
     * @param dt Шаг времени (не используется)
     * @param noise Флаг добавления шума
     * @return Eigen::Vector2d Вектор шума процесса размером 2
     */
    Eigen::VectorXd w(double t, double dt, bool noise = false)
    {
        if (!noise) {
            return Eigen::VectorXd::Zero(2);
        }
        // w(0) = 0, w(1) = коррелированный шум
        Eigen::Vector2d w_vec;
        w_vec << 0.0, generate_correlated_noise();
        return w_vec;
    }

    /**
     * @brief Шум измерений
     * @param t Время (для вычисления ковариации)
     * @param noise Флаг добавления шума
     * @return Eigen::Vector2d Вектор шума измерений размером 2
     */
    Eigen::VectorXd v(double t,
                      bool noise = false)
    {
        if (!noise) {
            return Eigen::Vector2d::Zero();
        }
        return kalman_noise::noise_gen.noiseWithCovariance(R(t));
    }

    // ------------------------------------------------------------------------
    // ДИНАМИКА СИСТЕМЫ И ИЗМЕРЕНИЯ
    // ------------------------------------------------------------------------

    /**
     * @brief Истинная динамика системы
     * @param x Текущее состояние [φ, p]ᵀ
     * @param t Текущее время (секунды)
     * @param dt Шаг времени (секунды)
     * @param scenario Сценарий управления
     * @param add_noise Флаг добавления шума процесса
     * @return Eigen::Vector2d Следующее состояние системы
     */
    Eigen::Vector2d true_dynamics(const Eigen::Vector2d& x,
                                  double t,
                                  double dt,
                                  ControlScenario scenario,
                                  bool add_noise = true)
    {
        Eigen::Vector2d x_next = A(dt) * x + D(dt) * u(t, scenario);
        if (add_noise) {
            Eigen::VectorXd w_noise = w(t, dt, true);
            x_next += B(dt) * w_noise(1);
        }
        return x_next;
    }

    /**
 * @brief Функция измерений (точная, нелинейная)
 * @param x Текущее состояние [φ, p]ᵀ
 * @param t Текущее время (для вычисления шума)
 * @param add_noise Флаг добавления шума измерений
 * @return Eigen::Vector2d Вектор измерений
 */
    Eigen::Vector2d measurement_exact(const Eigen::Vector2d& x,
                                      double t,
                                      bool add_noise = true)
    {
        Eigen::Vector2d y;
        y << x(1),                     // p (угловая скорость)
                g * std::sin(x(0));       // g·sin(φ) (точное)

        if (add_noise) {
            y += v(t, true);
        }

        return y;
    }

    /**
     * @brief Функция измерений (линеаризованная для фильтра Калмана)
     * @param x Текущее состояние [φ, p]ᵀ
     * @param t Текущее время (для вычисления шума)
     * @param add_noise Флаг добавления шума измерений
     * @return Eigen::Vector2d Вектор измерений
     */
    Eigen::Vector2d measurement_linearized(const Eigen::Vector2d& x,
                                           double t,
                                           bool add_noise = true)
    {
        Eigen::Vector2d y;
        y << x(1),                     // p (угловая скорость)
                g * x(0);                 // g·φ (линеаризованное)

        if (add_noise) {
            y += v(t, true);
        }

        return y;
    }
} // namespace model2

#endif // MODELS_HPP