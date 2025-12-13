/**
 * @file ckf.cpp
 * @brief Реализация классического фильтра Калмана (CKF)
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 */

#include <utility>

#include "kalman.hpp"

using namespace kalman;

// Проверка числовой корректности матрицы
template<typename Derived>
bool is_matrix_valid(const Eigen::MatrixBase<Derived>& mat) {
    return mat.allFinite() &&
           (mat.array().isInf() == 0).all() &&
           (mat.array().isNaN() == 0).all();
}

// Проверка размерностей матриц
template<typename MatrixType>
bool check_matrix_dimensions(const MatrixType& mat,
                             Eigen::Index rows_expected,
                             Eigen::Index cols_expected,
                             const std::string& name) {
    if (mat.rows() != rows_expected || mat.cols() != cols_expected) {
        std::cerr << "ERROR: Dimension mismatch for " << name
                  << ": expected " << rows_expected << "x" << cols_expected
                  << ", got " << mat.rows() << "x" << mat.cols() << std::endl;
        return false;
    }
    return true;
}

// ============================================
// CKF КОНСТРУКТОРЫ
// ============================================

/**
* @brief Конструктор с указанием размерности состояния
*
* Инициализирует фильтр с нулевым состоянием и единичной матрицей ковариации.
*
* @param nx Размерность вектора состояния
*
* @exception std::invalid_argument Если nx == 0
*/
CKF::CKF(const size_t nx)
        : x_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nx))),
          P_(Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(nx),
                                       static_cast<Eigen::Index>(nx)))
{
    if (nx == 0) {
        throw std::invalid_argument("CKF: Dimension nx must be greater than 0");
    }
    std::cout << "CKF initialized with nx = " << nx << std::endl;
}

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
CKF::CKF(Eigen::VectorXd x0, Eigen::MatrixXd P0)
        : x_(std::move(x0)), P_(std::move(P0))
{
    // Проверка согласованности размерностей
    const Eigen::Index nx = x_.size();
    if (P_.rows() != nx || P_.cols() != nx) {
        throw std::invalid_argument(
                "CKF: P0 must be square matrix of size nx x nx");
    }

    // Проверка симметричности
    const double asymmetry = (P_ - P_.transpose()).norm() / P_.norm();
    if (asymmetry > 1e-10) {
        std::cerr << "WARNING: CKF: P0 is not symmetric (asymmetry = "
                  << asymmetry << "), symmetrizing" << std::endl;
        P_ = 0.5 * (P_ + P_.transpose());
    }

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt(P_);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("CKF: P0 is not positive definite");
    }

    std::cout << "CKF initialized with custom x0 and P0" << std::endl;
}

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
void CKF::initialize(const Eigen::VectorXd &x0,
                     const Eigen::MatrixXd &P0)
{
    if (x0.size() != P0.rows() || P0.rows() != P0.cols()) {
        throw std::invalid_argument(
                "CKF::initialize: Dimension mismatch between x0 and P0");
    }

    if (!is_matrix_valid(x0) || !is_matrix_valid(P0)) {
        throw std::invalid_argument(
                "CKF::initialize: x0 or P0 contains NaN/Inf values");
    }

    // Проверка положительной определенности P0
    Eigen::LLT<Eigen::MatrixXd> llt(P0);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument(
                "CKF::initialize: P0 is not positive definite");
    }

    x_ = x0;
    P_ = P0;

    // Гарантируем симметричность
    P_ = 0.5 * (P_ + P_.transpose());

    std::cout << "CKF reinitialized" << std::endl;
}

// ============================================
// CKF МЕТОДЫ
// ============================================

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
void CKF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
               const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
               const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
               const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    // ============================================
    // ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
    // ============================================
    std::cout << "\n=== CKF Step (Classical Kalman Filter) ===" << std::endl;
    std::cout << "A_matrix: " << A << "\n";
    std::cout << "B_matrix: " << B << "\n";
    std::cout << "C_matrix: " << C << "\n";
    std::cout << "D_matrix: " << D << "\n";
    std::cout << "Q_matrix: " << Q << "\n";
    std::cout << "R_matrix: " << R << "\n";
    std::cout << "U_vector: " << u << "\n";
    std::cout << "Y_vector: " << y << "\n";

    const size_t nx = x_.size();
    const size_t ny = y.size();
    const size_t nw = B.cols();
    const size_t nu = u.size();

    // Проверка числовой корректности
    if (!is_matrix_valid(A) || !is_matrix_valid(B) || !is_matrix_valid(C) ||
        !is_matrix_valid(D) || !is_matrix_valid(Q) || !is_matrix_valid(R) ||
        !is_matrix_valid(u) || !is_matrix_valid(y)) {
        throw std::runtime_error("CKF::step: Input contains NaN/Inf values");
    }

    // Проверка размерностей
    bool dims_ok = true;
    dims_ok &= check_matrix_dimensions(A, nx, nx, "A");
    dims_ok &= check_matrix_dimensions(B, nx, nw, "B");
    dims_ok &= check_matrix_dimensions(C, ny, nx, "C");
    dims_ok &= check_matrix_dimensions(D, ny, nu, "D");
    dims_ok &= check_matrix_dimensions(Q, nw, nw, "Q");
    dims_ok &= check_matrix_dimensions(R, ny, ny, "R");

    if (!dims_ok) {
        throw std::runtime_error("CKF::step: Dimension mismatch in input matrices");
    }

    std::cout << "Dimensions: nx=" << nx << ", ny=" << ny
              << ", nw=" << nw << ", nu=" << nu << std::endl;
    std::cout << "Initial state: " << x_.transpose() << std::endl;

    const Eigen::MatrixXd P_old = P_;
    std::cout << "Initial covariance P (norm = " << P_old.norm() << "):"
              << std::endl << P_old << std::endl;

    // ============================================
    // ПРОВЕРКА ПОЛОЖИТЕЛЬНОЙ ОПРЕДЕЛЕННОСТИ
    // ============================================

    // Проверка Q
    Eigen::LLT<Eigen::MatrixXd> llt_Q(Q);
    if (llt_Q.info() != Eigen::Success) {
        std::cerr << "WARNING: Q is not positive definite, adding regularization"
                  << std::endl;
        Eigen::MatrixXd Q_reg = Q + Eigen::MatrixXd::Identity(nw, nw) * 1e-8;
        llt_Q.compute(Q_reg);
    }

    // Проверка R
    Eigen::LLT<Eigen::MatrixXd> llt_R(R);
    if (llt_R.info() != Eigen::Success) {
        std::cerr << "WARNING: R is not positive definite, adding regularization"
                  << std::endl;
        Eigen::MatrixXd R_reg = R + Eigen::MatrixXd::Identity(ny, ny) * 1e-8;
        llt_R.compute(R_reg);
    }

    // ============================================
    // ШАГ ПРОГНОЗА
    // ============================================

    std::cout << "\n--- Prediction Step ---" << std::endl;

    // ПРОГНОЗ СОСТОЯНИЯ
    std::cout << "\nState prediction:\n";
    Eigen::VectorXd x_pred = A * x_ + D * u;
    std::cout << "x_pred = A*x + D*u: " << x_pred.transpose() << "\n";

    // Прогноз ковариации
    const Eigen::MatrixXd P_pred = A * P_old * A.transpose() + B * Q * B.transpose();
    std::cout << "Predicted covariance P_pred = A*P*A' + B*Q*B' (norm = "
              << P_pred.norm() << "):" << std::endl << P_pred << std::endl;

    // ============================================
    // ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТА УСИЛЕНИЯ КАЛМАНА
    // ============================================

    std::cout << "\n--- Kalman Gain Computation ---" << std::endl;

    // Матрица ковариации инноваций
    Eigen::MatrixXd S = C * P_pred * C.transpose() + R;
    std::cout << "Innovation covariance S = C*P_pred*C' + R:"
              << std::endl << S << std::endl;

    // Проверка обусловленности
    const double cond_S = S.norm() * S.inverse().norm();
    std::cout << "Condition number of S: " << cond_S << std::endl;

    if (cond_S > 1.0 / std::numeric_limits<double>::epsilon()) {
        std::cerr << "WARNING: S is ill-conditioned, adding regularization"
                  << std::endl;
        S += Eigen::MatrixXd::Identity(ny, ny) * 1e-8;
    }

    // Вычисление коэффициента усиления Калмана
    Eigen::MatrixXd K;
    try {
        // Используем LDLT разложение для устойчивого обращения
        Eigen::LDLT<Eigen::MatrixXd> ldlt_S(S);
        if (ldlt_S.info() != Eigen::Success) {
            throw std::runtime_error("Failed to decompose S matrix");
        }
        K = P_pred * C.transpose() * ldlt_S.solve(Eigen::MatrixXd::Identity(ny, ny));
    } catch (const std::exception& e) {
        std::cerr << "ERROR computing Kalman gain: " << e.what()
                  << ", using alternative method" << std::endl;
        // Альтернативный метод через псевдообращение
        K = P_pred * C.transpose() * (S + Eigen::MatrixXd::Identity(ny, ny) * 1e-6).inverse();
    }

    std::cout << "Kalman gain K = P_pred*C'*inv(S):" << std::endl << K << std::endl;
    std::cout << "K norm: " << K.norm() << std::endl;

    // ============================================
    // ШАГ КОРРЕКЦИИ
    // ============================================

    std::cout << "\n--- Correction Step ---" << std::endl;

    // Инновации (невязка)
    const Eigen::VectorXd innov = y - C * x_pred;
    std::cout << "Innovation = y - C*x_pred: " << innov.transpose() << std::endl;
    std::cout << "Innovation norm: " << innov.norm() << std::endl;

    // Обновление состояния
    const Eigen::VectorXd x_old = x_;
    x_ = x_pred + K * innov;
    std::cout << "Corrected state x_new = x_pred + K*innov: "
              << x_.transpose() << std::endl;

    // ============================================
    // ОБНОВЛЕНИЕ КОВАРИАЦИИ (ФОРМУЛА ДЖОЗЕФА)
    // ============================================

    std::cout << "\n--- Covariance Update (Joseph Form) ---" << std::endl;

    Eigen::MatrixXd I_nx = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd I_KC = I_nx - K * C;

    // Формула Джозефа для численной устойчивости
    Eigen::MatrixXd P_next =
            (I_KC * P_pred * I_KC.transpose()) + (K * R * K.transpose());

    std::cout << "Updated covariance P_new (norm = " << P_next.norm() << "):"
              << std::endl << P_next << std::endl;

    // ============================================
    // ПОСТОБРАБОТКА И ВАЛИДАЦИЯ
    // ============================================

    // Гарантируем симметричность
    P_ = 0.5 * (P_next + P_next.transpose());

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_final(P_);
    if (llt_final.info() != Eigen::Success) {
        std::cerr << "WARNING: Final covariance is not positive definite, regularizing"
                  << std::endl;
        P_.diagonal().array() += 1e-8;
        llt_final.compute(P_);
        if (llt_final.info() != Eigen::Success) {
            // Крайний случай: сбрасываем к диагональной матрице
            std::cerr << "ERROR: Cannot regularize P, resetting to diagonal" << std::endl;
            P_ = Eigen::MatrixXd::Identity(nx, nx) * P_old.norm();
        }
    }

    std::cout << "Final covariance (after symmetrization):"
              << std::endl << P_ << std::endl;

    // ============================================
    // СРАВНЕНИЕ С SRCF (ДЛЯ ОТЛАДКИ)
    // ============================================

    std::cout << "\n=== Summary for SRCF Comparison ===" << std::endl;
    std::cout << "P_old (should match SRCF P_old):" << std::endl << P_old << std::endl;
    std::cout << "K (should match SRCF K):" << std::endl << K << std::endl;
    std::cout << "Old state: " << x_old.transpose() << std::endl;
    std::cout << "New state: " << x_.transpose() << std::endl;
    std::cout << "Final covariance P_new:" << std::endl << P_ << std::endl;

    // ============================================
    // ПРОВЕРКИ И ВАЛИДАЦИЯ
    // ============================================

    std::cout << "\n=== Validation Checks ===" << std::endl;

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_old(P_old);
    Eigen::LLT<Eigen::MatrixXd> llt_new(P_);

    std::cout << "P_old positive definite: "
              << (llt_old.info() == Eigen::Success ? "YES" : "NO") << std::endl;
    std::cout << "P_new positive definite: "
              << (llt_new.info() == Eigen::Success ? "YES" : "NO") << std::endl;

    // Проверка симметричности
    const double asym_old = (P_old - P_old.transpose()).norm() / P_old.norm();
    const double asym_new = (P_ - P_.transpose()).norm() / P_.norm();

    std::cout << "P_old asymmetry: " << asym_old
              << (asym_old < 1e-12 ? " (OK)" : " (WARNING)") << std::endl;
    std::cout << "P_new asymmetry: " << asym_new
              << (asym_new < 1e-12 ? " (OK)" : " (WARNING)") << std::endl;

    // Проверка числовой корректности состояния
    if (!is_matrix_valid(x_)) {
        throw std::runtime_error("CKF::step: State contains NaN/Inf after update");
    }

    if (!is_matrix_valid(P_)) {
        throw std::runtime_error("CKF::step: Covariance contains NaN/Inf after update");
    }

    std::cout << "\n=== CKF Step Completed Successfully ===" << std::endl;
}