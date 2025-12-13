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
    std::cout << "\n=== Manual CKF Step ===\n";
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

    std::cout << "nx=" << nx << ", ny=" << ny << ", nw=" << nw << ", nu=" << nu << "\n";
    std::cout << "Initial state: " << x_.transpose() << "\n";
    Eigen::MatrixXd P_old = P_;
    std::cout << "Initial covariance P_:\n" << P_old << "\n";

    // 1. ПРОГНОЗ КОВАРИАЦИИ
    std::cout << "\n1. Covariance prediction:\n";
    std::cout << "P_old Norm = " << P_.norm() << "\n";

    // 2. ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТА УСИЛЕНИЯ КАЛМАНА
    std::cout << "\n2. Kalman gain computation:\n";

    Eigen::MatrixXd S = C * P_old * C.transpose() + R;
    std::cout << "S = C*P*C' + R:\n" << S << "\n";
    std::cout << "Condition number of S: " << S.norm() * S.inverse().norm() << "\n";

    // Проверка S
    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if (llt_S.info() != Eigen::Success) {
        std::cout << "WARNING: S not positive definite!\n";
        S += Eigen::MatrixXd::Identity(ny, ny) * 1e-8;
    }

    Eigen::MatrixXd K = A * P_old * C.transpose() * S.inverse();
    std::cout << "K = P*C'*inv(S):\n" << K << "\n";
    std::cout << "K norm: " << K.norm() << "\n";

    // 3. ПРОГНОЗ СОСТОЯНИЯ
    std::cout << "\n3. State prediction:\n";
    Eigen::VectorXd x_pred = A * x_ + D * u;
    std::cout << "x_pred = A*x + D*u: " << x_pred.transpose() << "\n";

    // 4. КОРРЕКЦИЯ СОСТОЯНИЯ
    std::cout << "\n4. State correction:\n";
    Eigen::VectorXd innov = y - C * x_pred;
    std::cout << "Innovation = y - C*x_pred: " << innov.transpose() << "\n";
    std::cout << "Innovation norm: " << innov.norm() << "\n";

    Eigen::VectorXd x_corr = x_pred + K * innov;
    std::cout << "x_corr = x_pred + K*innov: " << x_corr.transpose() << "\n";
    Eigen::VectorXd x_old = x_;
    x_ = x_corr;

    // 5. КОРРЕКЦИЯ КОВАРИАЦИИ (ФОРМУЛА ДЖОЗЕФА)
    std::cout << "\n5. Covariance correction (Joseph form):\n";

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd I_KC = I - P_old * C.transpose() * S.inverse() * C;
    std::cout << "I - K*C:\n" << I_KC << "\n";

    Eigen::MatrixXd P_next = A * I_KC * P_old * A.transpose() + B * Q * B.transpose();
    std::cout << "P_next = A*I_KC*P*A' + B*Q*B':\n" << P_next << "\n";
    std::cout << "Norm: " << P_next.norm() << "\n";

    // Симметризация
    P_ = 0.5 * (P_next + P_next.transpose());
    std::cout << "After symmetrization:\n" << P_ << "\n";

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_final(P_);
    if (llt_final.info() != Eigen::Success) {
        std::cout << "WARNING: P_new not positive definite after update!\n";
        P_ = (P_ + P_.transpose()) / 2.0;
        P_.diagonal().array() += 1e-8;
    }

    // === СРАВНЕНИЕ С SRCF ===
    std::cout << "\n=== FOR SRCF COMPARISON (CKF Summary) ===\n";
    std::cout << "P_old (should match SRCF P_old):\n" << P_old << "\n";
    std::cout << "K (should match SRCF K = G*inv(S_Re)):\n" << K << "\n";
    std::cout << "Old x state: " << x_old.transpose() << "\n";
    std::cout << "Final new x state: " << x_.transpose() << "\n";
    std::cout << "Final covariance P_new (should match SRCF P_new):\n" << P_ << "\n";

    std::cout << "True state (A*[0,0] + B*u): " << (A * Eigen::Vector2d::Zero() + D * u).transpose() << "\n";

    // === ПРОВЕРКА ===
    std::cout << "\n=== CHECKS ===\n";

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_pred(P_old);
    Eigen::LLT<Eigen::MatrixXd> llt_new(P_);

    std::cout << "P_pred positive definite: "
              << (llt_pred.info() == Eigen::Success ? "YES" : "NO") << "\n";
    std::cout << "P_new positive definite: "
              << (llt_new.info() == Eigen::Success ? "YES" : "NO") << "\n";

    // Проверка симметрии
    double asym_pred = (P_old - P_old.transpose()).norm() / P_old.norm();
    double asym_new = (P_ - P_.transpose()).norm() / P_.norm();

    std::cout << "P_pred asymmetry: " << asym_pred << "\n";
    std::cout << "P_new asymmetry: " << asym_new << "\n";
}