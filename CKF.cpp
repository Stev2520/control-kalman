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
    std::cout << "Initial covariance P_:\n" << P_ << "\n";

    // 1. ПРОГНОЗ КОВАРИАЦИИ
    std::cout << "\n1. Covariance prediction:\n";
    std::cout << "P_old = " << P_.norm() << "\n";

    Eigen::MatrixXd P_pred = A * P_ * A.transpose() + B * Q * B.transpose();
    std::cout << "P_pred = A*P*A' + B*Q*B':\n" << P_pred << "\n";
    std::cout << "Norm: " << P_pred.norm() << "\n";

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_pred(P_pred);
    if (llt_pred.info() != Eigen::Success) {
        std::cout << "WARNING: P_pred not positive definite!\n";
        P_pred = (P_pred + P_pred.transpose()) / 2.0;
        P_pred.diagonal().array() += 1e-8;
    }

    P_ = P_pred;

    // 2. ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТА УСИЛЕНИЯ КАЛМАНА
    std::cout << "\n2. Kalman gain computation:\n";

    Eigen::MatrixXd S = C * P_ * C.transpose() + R;
    std::cout << "S = C*P*C' + R:\n" << S << "\n";
    std::cout << "Condition number of S: " << S.norm() * S.inverse().norm() << "\n";

    // Проверка S
    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if (llt_S.info() != Eigen::Success) {
        std::cout << "WARNING: S not positive definite!\n";
        S += Eigen::MatrixXd::Identity(ny, ny) * 1e-8;
    }

    Eigen::MatrixXd K = P_ * C.transpose() * S.inverse();
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

    x_ = x_corr;

    // 5. КОРРЕКЦИЯ КОВАРИАЦИИ (ФОРМУЛА ДЖОЗЕФА)
    std::cout << "\n5. Covariance correction (Joseph form):\n";

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd I_KC = I - K * C;
    std::cout << "I - K*C:\n" << I_KC << "\n";

    P_ = I_KC * P_ * I_KC.transpose() + K * R * K.transpose();
    std::cout << "P_new = (I-KC)*P*(I-KC)' + K*R*K':\n" << P_ << "\n";
    std::cout << "P_new norm: " << P_.norm() << "\n";

    // Симметризация
    P_ = 0.5 * (P_ + P_.transpose());
    std::cout << "After symmetrization:\n" << P_ << "\n";

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_final(P_);
    if (llt_final.info() != Eigen::Success) {
        std::cout << "WARNING: P_new not positive definite after update!\n";
        P_ = (P_ + P_.transpose()) / 2.0;
        P_.diagonal().array() += 1e-8;
    }

    std::cout << "\n=== CKF Summary ===\n";
    std::cout << "Final state: " << x_.transpose() << "\n";
    std::cout << "Final covariance:\n" << P_ << "\n";

    // Для сравнения: истинное состояние (если известно)
    std::cout << "True state (A*[0,0] + B*u): " << (A * Eigen::Vector2d::Zero() + B * u).transpose() << "\n";
}