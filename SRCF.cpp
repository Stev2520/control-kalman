/**
 * @file srcf.cpp
 * @brief Реализация квадратно-корневого фильтра Калмана (SRCF)
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * @note Использует QR-разложение для поддержания численной устойчивости
 *       и положительной определенности матрицы ковариации.
 */

#include <iostream>
#include <utility>
#include "kalman.hpp"

using namespace kalman;

// ============================================
// SRCF КОНСТРУКТОРЫ
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
SRCF::SRCF(const size_t nx)
        : x_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nx))),
          S_(Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(nx),
                                       static_cast<Eigen::Index>(nx)))
{
    if (nx == 0) {
        throw std::invalid_argument("SRCF: Dimension nx must be greater than 0");
    }
    std::cout << "SRCF initialized with nx = " << nx << std::endl;
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
 *
 * @note Матрица P0 разлагается на квадратный корень: S₀ = chol(P₀)
 */
SRCF::SRCF(Eigen::VectorXd x0, const Eigen::MatrixXd& P0)
        : x_(std::move(x0))
{
    // Проверка согласованности размерностей
    const Eigen::Index nx = x_.size();
    if (P0.rows() != nx || P0.cols() != nx) {
        throw std::invalid_argument(
                "SRCF: P0 must be square matrix of size nx x nx");
    }

    if (!P0.allFinite()) {
        throw std::invalid_argument("SRCF: P0 contains NaN/Inf values");
    }

    // Вычисление квадратного корня через разложение Холецкого
    Eigen::LLT<Eigen::MatrixXd> llt(P0);
    if (llt.info() != Eigen::Success) {
        std::cerr << "WARNING: SRCF: P0 is not positive definite, regularizing"
                  << std::endl;
        Eigen::MatrixXd P0_reg = P0 +
                                 Eigen::MatrixXd::Identity(nx, nx) * 1e-8;
        llt.compute(P0_reg);

        if (llt.info() != Eigen::Success) {
            throw std::invalid_argument(
                    "SRCF: Cannot compute Cholesky decomposition of P0");
        }
    }
    S_ = llt.matrixL();
    std::cout << "SRCF initialized with custom x0 and P0" << std::endl;
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
void SRCF::initialize(const Eigen::VectorXd& x0,
                      const Eigen::MatrixXd& P0)
{
    if (x0.size() != P0.rows() || P0.rows() != P0.cols()) {
        throw std::invalid_argument(
                "SRCF::initialize: Dimension mismatch between x0 and P0");
    }

    if (!x0.allFinite() || !P0.allFinite()) {
        throw std::invalid_argument(
                "SRCF::initialize: x0 or P0 contains NaN/Inf values");
    }

    // Вычисление квадратного корня через разложение Холецкого
    Eigen::LLT<Eigen::MatrixXd> llt(P0);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument(
                "SRCF::initialize: P0 is not positive definite");
    }

    x_ = x0;
    S_ = llt.matrixL();

    std::cout << "SRCF reinitialized" << std::endl;
}

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
void SRCF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
                const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    // ============================================
    // ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
    // ============================================
    std::cout << "\n=== SRCF Step (Square-Root Covariance Filter) ===" << std::endl;
    std::cout << "A_matrix: " << A << "\n";
    std::cout << "B_matrix: " << B << "\n";
    std::cout << "C_matrix: " << C << "\n";
    std::cout << "D_matrix: " << D << "\n";
    std::cout << "Q_matrix: " << Q << "\n";
    std::cout << "R_matrix: " << R << "\n";
    std::cout << "U_vector: " << u << "\n";
    std::cout << "Y_vector: " << y << "\n";

    const Eigen::Index nx = x_.size();
    const Eigen::Index ny = y.size();
    const Eigen::Index nw = B.cols();
    const Eigen::Index nu = u.size();
    std::cout << "nx=" << nx << ", ny=" << ny << ", nw=" << nw  << ", nu=" << nu << "\n";

    if (!A.allFinite() || !B.allFinite() || !C.allFinite() || !D.allFinite() ||
        !Q.allFinite() || !R.allFinite() || !u.allFinite() || !y.allFinite()) {
        std::cerr << "A finite: " << A.allFinite()
                  << ", B finite: " << B.allFinite()
                  << ", C finite: " << C.allFinite()
                  << ", Q finite: " << Q.allFinite()
                  << ", R finite: " << R.allFinite()
                  << ", u finite: " << u.allFinite()
                  << ", y finite: " << y.allFinite() << std::endl;
        throw std::runtime_error("SRCF::step: Input contains NaN/Inf values");
    }

    if (!(A.rows() == nx && A.cols() == nx &&
          B.rows() == nx && B.cols() == nw &&
          C.rows() == ny && C.cols() == nx &&
          D.rows() == nx && D.cols() == nu &&
          Q.rows() == nw && Q.cols() == nw &&
          R.rows() == ny && R.cols() == ny)) {
        std::stringstream ss;
        ss << "SRCF::step: Dimension mismatch: "
           << "A: " << A.rows() << "x" << A.cols() << " (expected " << nx << "x" << nx << "), "
           << "B: " << B.rows() << "x" << B.cols() << " (expected " << nx << "x" << nw << "), "
           << "C: " << C.rows() << "x" << C.cols() << " (expected " << ny << "x" << nx << "), "
           << "D: " << D.rows() << "x" << D.cols() << " (expected " << ny << "x" << nu << "), "
           << "Q: " << Q.rows() << "x" << Q.cols() << " (expected " << nw << "x" << nw << "), "
           << "R: " << R.rows() << "x" << R.cols() << " (expected " << ny << "x" << ny << ")";
        throw std::runtime_error(ss.str());
    }
    std::cout << "Dimensions: nx=" << nx << ", ny=" << ny
              << ", nw=" << nw << ", nu=" << nu << std::endl;

    // ============================================
    // ПОДГОТОВКА КВАДРАТНЫХ КОРНЕЙ
    // ============================================
    std::cout << "\n--- Cholesky Decompositions ---" << std::endl;

    // Квадратные корни матриц Q и R
    Eigen::MatrixXd SQ, SR;

    // Q decomposition
    Eigen::LLT<Eigen::MatrixXd> lltQ(Q);
    if (lltQ.info() != Eigen::Success) {
        std::cerr << "WARNING: Q is not positive definite! Adding regularization." << std::endl;
        Eigen::MatrixXd Q_reg = Q + Eigen::MatrixXd::Identity(Q.rows(), Q.cols()) * 1e-8;
        SQ = Q_reg.llt().matrixL();
    } else {
        SQ = Q.llt().matrixL();
    }

    // R decomposition
    Eigen::LLT<Eigen::MatrixXd> lltR(R);
    if (lltR.info() != Eigen::Success) {
        std::cerr << "WARNING: R is not positive definite! Adding regularization." << std::endl;
        Eigen::MatrixXd R_reg = R + Eigen::MatrixXd::Identity(R.rows(), R.cols()) * 1e-8;
        SR = R_reg.llt().matrixL();
    } else {
        SR = R.llt().matrixL();
    }

    std::cout << "SQ (" << SQ.rows() << "x" << SQ.cols() << "): " << SQ << "\n";
    std::cout << "SR (" << SR.rows() << "x" << SR.cols() << "): " << SR << "\n";

    // Проверка текущего S_
    std::cout << "Initial S_: " << S_ << "\n";
    Eigen::MatrixXd P_old = S_ * S_.transpose();
    std::cout << "Initial covariance P_: " << P_old << "\n";
    std::cout << "\nCovariance P_old prediction:\n";
    std::cout << "P_old Norm = " << P_old.norm() << "\n";
    if (!S_.allFinite()) {
        std::cerr << "WARNING: S_ contains NaN/Inf, resetting to identity" << std::endl;
        S_ = Eigen::MatrixXd::Identity(nx, nx) * 0.1;
    }

    // ============================================
    // ФОРМИРОВАНИЕ ПРЕАРРЕЯ ДЛЯ QR-РАЗЛОЖЕНИЯ
    // ============================================
    std::cout << "\n--- Building Prearray for QR Decomposition ---" << std::endl;

    // Вычисление промежуточных матриц
    Eigen::MatrixXd C_times_S = C * S_;
    Eigen::MatrixXd A_times_S = A * S_;
    Eigen::MatrixXd B_times_SQ = B * SQ;

    std::cout << "C*S (" << C_times_S.rows() << "x" << C_times_S.cols() << "):\n" << C_times_S << "\n";
    std::cout << "A*S (" << A_times_S.rows() << "x" << A_times_S.cols() << "):\n" << A_times_S << "\n";
    std::cout << "B*SQ (" << B_times_SQ.rows() << "x" << B_times_SQ.cols() << "):\n" << B_times_SQ << "\n";

    // Формирование преаррея размера (nx+ny) x (nx+ny+nw)
    std::cout << "\nBuilding prearray (" << (nx + ny) << "x" << (nx + ny + nw) << "):\n";
    Eigen::MatrixXd prearray = Eigen::MatrixXd::Zero(nx + ny, nx + ny + nw);

    try {
        // Верхний блок: [SR, C*S, 0]
        prearray.block(0, 0, ny, ny) = SR;
        prearray.block(0, ny, ny, nx) = C_times_S;
        prearray.block(0, nx + ny, ny, nw).setZero();;

        // Нижний блок: [0, A*S, B*SQ]
        prearray.block(ny, 0, nx, ny).setZero();
        prearray.block(ny, ny, nx, nx) = A_times_S;
        prearray.block(ny, nx + ny, nx, nw) = B_times_SQ;
    } catch (const std::exception& e) {
        std::cerr << "ERROR forming prearray: " << e.what() << std::endl;
        return;
    }

    std::cout << "Prearray (" << prearray.rows() << "x" << prearray.cols() << "):\n"
              << prearray << "\n";

    if (!prearray.allFinite()) {
        std::cout << "SR norm: " << SR.norm() << std::endl;
        std::cout << "C*S_ norm: " << (C * S_).norm() << std::endl;
        std::cout << "A*S_ norm: " << (A * S_).norm() << std::endl;
        std::cout << "B*SQ norm: " << (B * SQ).norm() << std::endl;
        throw std::runtime_error("SRCF::step: Prearray contains NaN/Inf");
    }

    // ============================================
    // QR-РАЗЛОЖЕНИЕ
    // ============================================
    std::cout << "\n--- QR Decomposition ---" << std::endl;
    Eigen::MatrixXd R_mat, Q_mat;
    try {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(prearray.transpose());
        R_mat = qr.matrixQR().template triangularView<Eigen::Upper>();
        if (R_mat.rows() > nx + ny) {
            R_mat = R_mat.topLeftCorner(nx + ny, nx + ny);
        }
        std::cout << "R_mat (" << R_mat.rows() << "x" << R_mat.cols() << "):\n" << R_mat << "\n";
        Q_mat = qr.matrixQR().template triangularView<Eigen::Lower>();
        std::cout << "Q_mat (" << Q_mat.rows() << "x" << Q_mat.cols() << "):\n" << Q_mat << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR in QR decomposition: " << e.what() << std::endl;
        throw std::runtime_error("SRCF::step: Failed QR decomposition");
    }

    if (!R_mat.allFinite()) {
        throw std::runtime_error("SRCF::step: R_mat contains NaN/Inf after QR");
    }

    if (!Q_mat.allFinite()) {
        throw std::runtime_error("SRCF::step: Q_mat contains NaN/Inf after QR");
    }

    // ============================================
    // ИЗВЛЕЧЕНИЕ БЛОКОВ ПОСЛЕ QR
    // ============================================
    std::cout << "\n--- Extracting Postarray Blocks ---" << std::endl;

    // Извлечение постаррея (транспонирование R)
    Eigen::MatrixXd postarray = -R_mat.transpose();
    std::cout << "\nPostarray (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";

    if (!postarray.allFinite()) {
        std::cout << "ERROR: postarray contains NaN/Inf!" << std::endl;
        return;
    }

    if (postarray.rows() < ny + nx || postarray.cols() < ny + nx) {
        std::cout << "ERROR: postarray too small: "
                  << postarray.rows() << "x" << postarray.cols()
                  << ", expected at least " << ny + nx << "x" << ny + nx << std::endl;
        return;
    }

    // Коррекция знаков для положительных диагональных элементов
    for (Eigen::Index i = 0; i < ny; ++i) {
        if (postarray(i, i) < 0) {
            postarray.row(i) = -postarray.row(i);
        }
    }

    for (Eigen::Index i = 0; i < nx; ++i) {
        Eigen::Index row_idx = ny + i;
        if (postarray(row_idx, row_idx) < 0) {
            postarray.row(row_idx) = -postarray.row(row_idx);
        }
    }
    std::cout << "\nPostarray after sign update (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";

    // Извлекаем блоки
    // post = [S_Re    0]
    //        [G      S_next]
    std::cout << "\nExtracting blocks:\n";
    Eigen::MatrixXd S_Re = postarray.block(0, 0, ny, ny);
    Eigen::MatrixXd G = postarray.block(ny, 0, nx, ny);
    Eigen::MatrixXd S_next = postarray.block(ny, ny, nx, nx);

    std::cout << "\nExtracted blocks:" << std::endl;
    std::cout << "S_Re (square root of innovation covariance):"
              << std::endl << S_Re << std::endl;
    std::cout << "G (gain-related matrix):" << std::endl << G << std::endl;
    std::cout << "S_next (square root of predicted covariance):"
              << std::endl << S_next << std::endl;

    if (!S_next.allFinite()) {
        std::cerr << "WARNING: S_next contains NaN/Inf, using fallback" << std::endl;
        return;
    }

    // ============================================
    // ВЫЧИСЛЕНИЕ ЭКВИВАЛЕНТНЫХ ВЕЛИЧИН
    // ============================================
    std::cout << "\n--- Computing Equivalent Quantities ---" << std::endl;

    // P_next из SRCF (должно совпадать с CKF P_next)
    Eigen::MatrixXd P_next_srcf = S_next * S_next.transpose();
    std::cout << "SRCF P_next = S_next * S_next^T:\n" << P_next_srcf << std::endl;

    // Калманово усиление из SRCF: K = G * inv(S_Re)
    Eigen::MatrixXd K_srcf;
    try {
        // S_Re должна быть нижнетреугольной
        K_srcf = G * S_Re.triangularView<Eigen::Lower>().solve(
                Eigen::MatrixXd::Identity(ny, ny));
        std::cout << "SRCF K = G * inv(S_Re):" << std::endl << K_srcf << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR computing Kalman gain: " << e.what() << std::endl;
        K_srcf = G * (S_Re + Eigen::MatrixXd::Identity(ny, ny) * 1e-8).inverse();
    }

    // S матрица ковариации инноваций: S = S_Re * S_Re^T
    Eigen::MatrixXd S_srcf = S_Re * S_Re.transpose();
    std::cout << "SRCF S = S_Re * S_Re^T (innovation covariance):"
              << std::endl << S_srcf << std::endl;

    std::cout << "Condition number of S_Re: "
              << S_Re.norm() * S_Re.inverse().norm() << "\n";

    // ============================================
    // ОБНОВЛЕНИЕ СОСТОЯНИЯ
    // ============================================
    std::cout << "\n--- State Update ---" << std::endl;

    // Прогноз состояния
    Eigen::VectorXd x_pred = A * x_ + D * u;
    std::cout << "Predicted state x_pred = A*x + D*u: "
              << x_pred.transpose() << std::endl;

    // Инновации
    Eigen::VectorXd innov = y - C * x_pred;
    std::cout << "Innovation = y - C*x_pred: " << innov.transpose() << std::endl;

    // Решение треугольной системы S_Re * z = innov
    Eigen::VectorXd z;
    try {
        z = S_Re.triangularView<Eigen::Lower>().solve(innov);
        std::cout << "z = S_Re\\innov: " << z.transpose() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << ", using zero innovation" << std::endl;
        z = Eigen::VectorXd::Zero(ny);
    }

    // Обновление состояния
    Eigen::VectorXd x_old = x_;
    x_ = x_pred + G * z;
    std::cout << "x_old: " << x_old << " \n";
    std::cout << "Updated state x_new = x_pred + G*z: "
              << x_.transpose() << std::endl;
    std::cout << "  A*x = " << (A * x_).transpose() << "\n";
    std::cout << "  G*z = " << (G * z).transpose() << "\n";
    std::cout << "  D*u = " << (D * u).transpose() << "\n";

    if (!z.allFinite()) {
        std::cerr << "WARNING: z contains NaN/Inf, using zero update" << std::endl;
        x_ = x_pred;
    }

    // ============================================
    // ОБНОВЛЕНИЕ КОВАРИАЦИИ И ВАЛИДАЦИЯ
    // ============================================

    // Обновление квадратного корня ковариации
    S_ = S_next;

    // Гарантируем нижнетреугольность
    S_ = S_.triangularView<Eigen::Lower>().toDenseMatrix();

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_test(P_next_srcf);
    if (llt_test.info() != Eigen::Success) {
        std::cerr << "WARNING: P not positive definite after update, regularizing" << std::endl;
        // Регуляризация
        P_next_srcf += Eigen::MatrixXd::Identity(nx, nx) * 1e-8;
        S_ = P_next_srcf.llt().matrixL();
    }

    // ============================================
    // СРАВНЕНИЕ С CKF (ДЛЯ ОТЛАДКИ)
    // ============================================
    std::cout << "\n=== FOR CKF COMPARISON (SRCF Summary) ===\n";
    std::cout << "S_Re (should be Cholesky factor of S):\n" << S_Re << "\n";
    std::cout << "S_next (Cholesky factor of P_pred):\n" << S_next << "\n";
    std::cout << "P_old (should match CKF P_old):\n" << P_old << "\n";
    std::cout << "K (should match CKF K = G*inv(S_Re)):\n" << G << "\n";
    std::cout << "Old x state: " << x_old.transpose() << "\n";
    std::cout << "Final new x state: " << x_.transpose() << "\n";
    std::cout << "Final covariance P_new (should match SKF P_new):\n" << P_next_srcf << "\n";
    std::cout << "True state: " << (A * Eigen::Vector2d(0,0) + D * u).transpose() << "\n";

    // ============================================
    // ПРОВЕРКИ И ВАЛИДАЦИЯ
    // ============================================

    std::cout << "\n=== Validation Checks ===" << std::endl;
    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_old(P_old);
    Eigen::LLT<Eigen::MatrixXd> llt_new(P_next_srcf);

    std::cout << "P_old positive definite: "
              << (llt_old.info() == Eigen::Success ? "YES" : "NO") << std::endl;
    std::cout << "P_new positive definite: "
              << (llt_new.info() == Eigen::Success ? "YES" : "NO") << std::endl;

    // Проверка симметричности
    double asym_old = (P_old - P_old.transpose()).norm() / P_old.norm();
    double asym_new = (P_next_srcf - P_next_srcf.transpose()).norm() / P_next_srcf.norm();

    std::cout << "P_old asymmetry: " << asym_old
              << (asym_old < 1e-12 ? " (OK)" : " (WARNING)") << std::endl;
    std::cout << "P_new asymmetry: " << asym_new
              << (asym_new < 1e-12 ? " (OK)" : " (WARNING)") << std::endl;

    // Проверка числовой корректности
    if (!x_.allFinite()) {
        throw std::runtime_error("SRCF::step: State contains NaN/Inf after update");
    }

    if (!S_.allFinite()) {
        throw std::runtime_error("SRCF::step: Covariance sqrt contains NaN/Inf after update");
    }

    std::cout << "\n=== SRCF Step Completed Successfully ===" << std::endl;
}