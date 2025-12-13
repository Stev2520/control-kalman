/**
 * @file kalman.hpp
 * @brief Реализация фильтров Калмана (Классический и Квадратно-корневой)
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * Данный заголовочный файл содержит реализации двух вариантов фильтра Калмана:
 * 1. Классический фильтр Калмана (CKF)
 * 2. Квадратно-корневой фильтр Калмана (SRCF)
 *
 * Квадратно-корневая версия обеспечивает лучшую численную устойчивость
 * при работе с плохо обусловленными матрицами ковариации.
 */

#ifndef KALMAN_HPP
#define KALMAN_HPP

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <stdexcept>

/**
 * @namespace kalman
 * @brief Пространство имен для реализаций фильтра Калмана
 *
 * Содержит классы для фильтрации и оценки состояния динамических систем
 * на основе алгоритма Калмана.
 */
namespace kalman
{
/**
 * @class CKF
 * @brief Классический фильтр Калмана (Classical Kalman Filter)
 *
 * Реализует стандартный алгоритм фильтра Калмана для дискретных
 * линейных систем. Поддерживает прогноз и коррекцию состояния.
 *
 * Математическая модель системы:
 * - xₖ = Aₖxₖ₋₁ + Dₖuₖ + Bₖwₖ, wₖ ∼ N(0, Qₖ)
 * - yₖ = Cₖxₖ + vₖ, vₖ ∼ N(0, Rₖ)
 *
 * @note Для обеспечения численной устойчивости рекомендуется использовать
 *       квадратно-корневую версию (SRCF) при работе с плохо обусловленными
 *       матрицами ковариации.
 *
 * @see SRCF
 */
class CKF
{
public:
    /**
     * @brief Конструктор с указанием размерности состояния
     *
     * Инициализирует фильтр с нулевым состоянием и единичной матрицей ковариации.
     *
     * @param nx Размерность вектора состояния
     *
     * @exception std::invalid_argument Если nx == 0
     */
    explicit CKF(size_t nx);

    /**
     * @brief Конструктор с начальными условиями
     *
     * Инициализирует фильтр с заданными начальными состоянием и ковариацией.
     *
     * @param x0 Начальное состояние (вектор размерности nx)
     * @param P0 Начальная матрица ковариации (размером nx × nx)
     *
     * @exception std::invalid_argument Если размерности не согласованы
     * @exception std::invalid_argument Если P0 не положительно определена
     */
    CKF(Eigen::VectorXd x0, Eigen::MatrixXd P0);

    /**
     * @brief Инициализация фильтра
     *
     * Устанавливает начальное состояние и ковариацию фильтра.
     *
     * @param x0 Начальное состояние
     * @param P0 Начальная матрица ковариации
     *
     * @exception std::invalid_argument Если размерности не согласованы
     * @exception std::invalid_argument Если P0 не положительно определена
     */
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);

    /**
     * @brief Выполнение одного шага фильтрации
     *
     * Выполняет прогноз и коррекцию состояния на основе новых измерений.
     *
     * @param A Матрица перехода состояния (размером nx × nx)
     * @param B Матрица управления (размером nx × nu)
     * @param C Матрица измерений (размером ny × nx)
     * @param D Матрица прямой связи (размером ny × nu)
     * @param Q Матрица ковариации шума процесса (размером nx × nx)
     * @param R Матрица ковариации шума измерений (размером ny × ny)
     * @param u Вектор управления (размерности nu)
     * @param y Вектор измерений (размерности ny)
     *
     * @exception std::runtime_error Если матрицы имеют несовместимые размерности
     * @exception std::runtime_error Если матрицы не положительно определены
     *
     * @note Алгоритм шага:
     *       1. Прогноз: x̂ₖ⁻ = Aₖx̂ₖ₋₁ + Bₖuₖ
     *       2. Прогноз ковариации: Pₖ⁻ = AₖPₖ₋₁Aₖᵀ + Qₖ
     *       3. Коэффициент усиления: Kₖ = Pₖ⁻Cₖᵀ(CₖPₖ⁻Cₖᵀ + Rₖ)⁻¹
     *       4. Коррекция: x̂ₖ = x̂ₖ⁻ + Kₖ(yₖ - Cₖx̂ₖ⁻ - Dₖuₖ)
     *       5. Обновление ковариации: Pₖ = (I - KₖCₖ)Pₖ⁻
     */
    void step(const Eigen::MatrixXd &A,
              const Eigen::MatrixXd &B,
              const Eigen::MatrixXd &C,
              const Eigen::MatrixXd &D,
              const Eigen::MatrixXd &Q,
              const Eigen::MatrixXd &R,
              const Eigen::VectorXd &u,
              const Eigen::VectorXd &y);

    /**
     * @brief Получить текущее состояние
     *
     * @return const Eigen::VectorXd& Текущая оценка состояния
     */
    [[nodiscard]] const Eigen::VectorXd& state() const { return x_; }

    /**
     * @brief Получить текущую матрицу ковариации
     *
     * @return const Eigen::MatrixXd& Текущая матрица ковариации оценки
     */
    [[nodiscard]] const Eigen::MatrixXd& covariance() const { return P_; }
private:
    Eigen::VectorXd x_; /**< Текущая оценка состояния */
    Eigen::MatrixXd P_; /**< Текущая матрица ковариации оценки */
};

/**
 * @class SRCF
 * @brief Квадратно-корневой фильтр Калмана (Square-Root Covariance Filter)
 *
 * Реализует алгоритм фильтра Калмана, работающий с квадратным корнем
 * матрицы ковариации (S = chol(P)). Это обеспечивает лучшую численную
 * устойчивость, так как ковариация всегда остается положительно
 * определенной.
 *
 * Основное уравнение: P = S·Sᵀ, где S - нижняя треугольная матрица.
 *
 * Преимущества перед классическим фильтром:
 * 1. Лучшая численная устойчивость
 * 2. Автоматическое поддержание положительной определенности P
 * 3. Уменьшение ошибок округления
 *
 * @see CKF
 */
class SRCF
{
public:
    /**
     * @brief Конструктор с указанием размерности состояния
     *
     * Инициализирует фильтр с нулевым состоянием и единичной матрицей ковариации.
     *
     * @param nx Размерность вектора состояния
     *
     * @exception std::invalid_argument Если nx == 0
     */
    explicit SRCF(size_t nx);

    /**
     * @brief Конструктор с начальными условиями
     *
     * Инициализирует фильтр с заданными начальными состоянием и ковариацией.
     *
     * @param x0 Начальное состояние (вектор размерности nx)
     * @param P0 Начальная матрица ковариации (размером nx × nx)
     *
     * @exception std::invalid_argument Если размерности не согласованы
     * @exception std::invalid_argument Если P0 не положительно определена
     *
     * @note Матрица P0 разлагается на квадратный корень: S₀ = chol(P₀)
     */
    SRCF(Eigen::VectorXd x0, const Eigen::MatrixXd &P0);

    /**
     * @brief Инициализация фильтра
     *
     * Устанавливает начальное состояние и ковариацию фильтра.
     *
     * @param x0 Начальное состояние
     * @param P0 Начальная матрица ковариации
     *
     * @exception std::invalid_argument Если размерности не согласованы
     * @exception std::invalid_argument Если P0 не положительно определена
     */
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);

    /**
     * @brief Выполнение одного шага фильтрации
     *
     * Выполняет прогноз и коррекцию состояния на основе новых измерений
     * с использованием квадратно-корневой формы.
     *
     * @param A Матрица перехода состояния (размером nx × nx)
     * @param B Матрица управления (размером nx × nu)
     * @param C Матрица измерений (размером ny × nx)
     * @param D Матрица прямой связи (размером ny × nu)
     * @param Q Матрица ковариации шума процесса (размером nx × nx)
     * @param R Матрица ковариации шума измерений (размером ny × ny)
     * @param u Вектор управления (размерности nu)
     * @param y Вектор измерений (размерности ny)
     *
     * @exception std::runtime_error Если матрицы имеют несовместимые размерности
     * @exception std::runtime_error Если матрицы не положительно определены
     *
     * @note Использует QR-разложение для поддержания треугольной формы S
     * @note Реализация основана на алгоритме Potter
     */
    void step(const Eigen::MatrixXd &A,
              const Eigen::MatrixXd &B,
              const Eigen::MatrixXd &C,
              const Eigen::MatrixXd &D,
              const Eigen::MatrixXd &Q,
              const Eigen::MatrixXd &R,
              const Eigen::VectorXd &u,
              const Eigen::VectorXd &y);

    /**
     * @brief Получить текущее состояние
     *
     * @return const Eigen::VectorXd& Текущая оценка состояния
     */
    [[nodiscard]] const Eigen::VectorXd& state() const { return x_; }

    /**
     * @brief Получить текущую матрицу ковариации
     *
     * Вычисляет полную матрицу ковариации из квадратного корня: P = S·Sᵀ.
     * Включает проверку на корректность результата.
     *
     * @return Eigen::MatrixXd Текущая матрица ковариации оценки
     *
     * @warning Возвращает копию матрицы, не ссылку
     * @note В случае обнаружения NaN/Inf в вычислениях возвращает
     *       диагональную матрицу с малыми значениями для предотвращения
     *       сбоев в вызывающем коде.
     */
    [[nodiscard]] Eigen::MatrixXd covariance() const
    {
        try {
            Eigen::MatrixXd P = S_ * S_.transpose();
            if (!P.allFinite()) {
                std::cerr << "WARNING: covariance contains NaN/Inf, returning safe diagonal matrix"
                          << std::endl;
                return Eigen::MatrixXd::Identity(S_.rows(), S_.rows()) * 0.1;
            }

            Eigen::LLT<Eigen::MatrixXd> llt(P);
            if (llt.info() != Eigen::Success) {
                std::cerr << "WARNING: covariance is not positive definite, returning safe diagonal matrix"
                          << std::endl;
                return Eigen::MatrixXd::Identity(S_.rows(), S_.rows()) * 0.1;
            }

            return P;
        }
        catch (const std::exception& e) {
            std::cerr << "ERROR in covariance(): " << e.what()
                      << ", returning safe diagonal matrix" << std::endl;
            return Eigen::MatrixXd::Identity(S_.rows(), S_.rows()) * 0.1;
        }
    }

    /**
     * @brief Получить квадратный корень матрицы ковариации
     *
     * @return const Eigen::MatrixXd& Матрица S, где P = S·Sᵀ
     */
    [[nodiscard]] const Eigen::MatrixXd& covarianceSqrt() const { return S_; }
private:
    Eigen::VectorXd x_; /**< Текущая оценка состояния */
    Eigen::MatrixXd S_; /**< Квадратный корень матрицы ковариации (нижняя треугольная) */
};
} // namespace kalman

#endif // KALMAN_HPP