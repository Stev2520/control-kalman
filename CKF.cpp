#include <utility>

#include "kalman.hpp"

using namespace kalman;

CKF::CKF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)),
                            P_(Eigen::MatrixXd::Identity(nx, nx)) { }

CKF::CKF(Eigen::VectorXd x0,
         Eigen::MatrixXd P0) : x_(std::move(x0)),
                               P_(std::move(P0)) { }

void CKF::initialize(const Eigen::VectorXd &x0,
                     const Eigen::MatrixXd &P0)
{
    x_ = x0;
    P_ = P0;
}

void CKF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
               const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
               const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
               const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    const size_t nx = x_.size();
    const size_t ny = y.size();
    const size_t nw = B.cols();
    const size_t nu = u.size();

    if (!A.allFinite() || !B.allFinite() || !C.allFinite() ||
        !Q.allFinite() || !R.allFinite() || !u.allFinite() || !y.allFinite()) {
        std::cerr << "ERROR: Input matrices contain NaN/Inf!" << std::endl;
        std::cerr << "A finite: " << A.allFinite()
                  << ", B finite: " << B.allFinite()
                  << ", C finite: " << C.allFinite()
                  << ", Q finite: " << Q.allFinite()
                  << ", R finite: " << R.allFinite()
                  << ", u finite: " << u.allFinite()
                  << ", y finite: " << y.allFinite() << std::endl;
        return;
    }

    if (!(A.rows() == nx && A.cols() == nx &&
          B.rows() == nx && B.cols() == nw &&
          C.rows() == ny && C.cols() == nx &&
          D.rows() == nx && D.cols() == nu &&
          Q.rows() == nw && Q.cols() == nw &&
          R.rows() == ny && R.cols() == ny)) {
        std::cerr << "ERROR: Dimension mismatch!" << std::endl;
        std::cerr << "A: " << A.rows() << "x" << A.cols() << " (expected " << nx << "x" << nx << ")\n";
        std::cerr << "B: " << B.rows() << "x" << B.cols() << " (expected " << nx << "x" << nw << ")\n";
        std::cerr << "C: " << C.rows() << "x" << C.cols() << " (expected " << ny << "x" << nx << ")\n";
        std::cerr << "D: " << D.rows() << "x" << D.cols() << " (expected " << nx << "x" << nu << ")\n";
        std::cerr << "Q: " << Q.rows() << "x" << Q.cols() << " (expected " << nw << "x" << nw << ")\n";
        std::cerr << "R: " << R.rows() << "x" << R.cols() << " (expected " << ny << "x" << ny << ")\n";
        return;
    }

    if (!P_.allFinite()) {
        std::cerr << "WARNING: P_ contains NaN/Inf, resetting to identity" << std::endl;
        P_ = Eigen::MatrixXd::Identity(nx, nx) * 0.1;
    }

    // 1. ПРОГНОЗ КОВАРИАЦИИ (априорная)
    // P_ = A * P_ * A^T + B * Q * B^T
    Eigen::MatrixXd P_pred;
    try {
        P_pred = A * P_ * A.transpose() + B * Q * B.transpose();
    } catch (const std::exception& e) {
        std::cerr << "ERROR in covariance prediction: " << e.what() << std::endl;
        return;
    }

    if (!P_pred.allFinite()) {
        std::cerr << "ERROR: P_pred contains NaN/Inf after prediction!" << std::endl;
        // Запасной вариант: простой прогноз
        x_ = A * x_ + D * u;
        return;
    }

    Eigen::LLT<Eigen::MatrixXd> llt_pred(P_pred);
    if (llt_pred.info() != Eigen::Success) {
        std::cerr << "WARNING: P_pred not positive definite, regularizing" << std::endl;
        P_pred = (P_pred + P_pred.transpose()) / 2.0;  // Симметризация
        P_pred.diagonal().array() += 1e-8;  // Регуляризация
    }

    P_ = P_pred;

    // 2. ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТА УСИЛЕНИЯ КАЛМАНА
    // S = C * P_ * C^T + R
    // K = P_ * C^T * S^{-1}
    Eigen::MatrixXd S, K;
    try {
        // Используем LDLT вместо явного inverse для устойчивости
        S = C * P_ * C.transpose() + R;

        // Проверка S
        Eigen::LLT<Eigen::MatrixXd> llt_S(S);
        if (llt_S.info() != Eigen::Success) {
            std::cerr << "WARNING: S not positive definite, regularizing" << std::endl;
            S += Eigen::MatrixXd::Identity(ny, ny) * 1e-8;
        }

        // Решаем S * Kᵀ = C * P_ (более устойчиво чем S.inverse())
        K = S.ldlt().solve(C * P_).transpose();
//        K = P_ * C.transpose() * S.inverse();

    } catch (const std::exception& e) {
        std::cerr << "ERROR computing Kalman gain: " << e.what() << std::endl;
        // Запасной вариант: нулевое усиление
        x_ = A * x_ + D * u;
        return;
    }

    if (!K.allFinite()) {
        std::cerr << "ERROR: K contains NaN/Inf!" << std::endl;
        x_ = A * x_ + D * u;
        return;
    }

    // 3. ПРОГНОЗ СОСТОЯНИЯ
    // x_ = A * x_ + D * u
    Eigen::VectorXd x_pred;
    try {
        x_pred = A * x_ + D * u;
    } catch (const std::exception& e) {
        std::cerr << "ERROR in state prediction: " << e.what() << std::endl;
        return;
    }

    if (!x_pred.allFinite()) {
        std::cerr << "ERROR: x_pred contains NaN/Inf!" << std::endl;
        return;
    }

    // 4. КОРРЕКЦИЯ СОСТОЯНИЯ (апостериорная)
    // x_ = x_ + K * (y - C * x_)
    Eigen::VectorXd innov;
    try {
        innov = y - C * x_pred;
    } catch (const std::exception& e) {
        std::cerr << "ERROR computing innovation: " << e.what() << std::endl;
        x_ = x_pred;  // Используем только прогноз
        return;
    }

    // Проверка слишком больших инноваций
    double innov_norm = innov.norm();
//    if (innov_norm > 100.0) {
//        std::cerr << "WARNING: Large innovation (norm = " << innov_norm << "), limiting update" << std::endl;
//        innov = innov / innov_norm * 100.0;
//    }

    try {
        x_ = x_pred + K * innov;
    } catch (const std::exception& e) {
        std::cerr << "ERROR in state correction: " << e.what() << std::endl;
        x_ = x_pred;  // Используем только прогноз
    }

    // 5. КОРРЕКЦИЯ КОВАРИАЦИИ (ФОРМУЛА ДЖОЗЕФА - устойчивая)
    // P_ = (I - K*C) * P_ * (I - K*C)^T + K * R * K^T
    try {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);
        Eigen::MatrixXd I_KC = I - K * C;

        // Упрощенная формула Джозефа для эффективности
        P_ = I_KC * P_ * I_KC.transpose() + K * R * K.transpose();

    } catch (const std::exception& e) {
        std::cerr << "ERROR in covariance correction: " << e.what() << std::endl;
        // Оставляем P_ как есть (прогноз)
        return;
    }

    // 8. ГАРАНТИРОВАННАЯ СИММЕТРИЗАЦИЯ И РЕГУЛЯРИЗАЦИЯ
    if (!P_.allFinite()) {
        std::cerr << "ERROR: P_ contains NaN/Inf after correction!" << std::endl;
        P_ = Eigen::MatrixXd::Identity(nx, nx) * 0.1;
        return;
    }

    // Симметризация для численной устойчивости (рекомендация из статьи)
    P_ = 0.5 * (P_ + P_.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt_final(P_);
    if (llt_final.info() != Eigen::Success) {
        std::cerr << "WARNING: P_ not positive definite after update, regularizing" << std::endl;
        P_ = (P_ + P_.transpose()) / 2.0;  // Симметризация
        P_.diagonal().array() += 1e-8;     // Регуляризация
    }

    // 9. ФИНАЛЬНАЯ ПРОВЕРКА
    if (!x_.allFinite() || !P_.allFinite()) {
        std::cerr << "CRITICAL ERROR: State or covariance corrupted!" << std::endl;
        std::cerr << "x_ finite: " << x_.allFinite()
                  << ", P_ finite: " << P_.allFinite() << std::endl;

        // Аварийный сброс
        x_ = Eigen::VectorXd::Zero(nx);
        P_ = Eigen::MatrixXd::Identity(nx, nx) * 0.1;
    }
}