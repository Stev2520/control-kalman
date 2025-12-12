#pragma once
#include <Eigen/Dense>
#include <functional>
#include <iostream>

namespace kalman
{
class CKF
{
public:
    CKF(const size_t nx);
    CKF(Eigen::VectorXd x0, Eigen::MatrixXd P0);
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y);
    const Eigen::VectorXd& state() const { return x_; }
    const Eigen::MatrixXd& covariance() const { return P_; }
private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
};
class SRCF
{
public:
    SRCF(const size_t nx);
    SRCF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y);
    const Eigen::VectorXd& state() const { return x_; }
    Eigen::MatrixXd& covariance() const {
        static Eigen::MatrixXd safe_cov;
        try {
            safe_cov = S_ * S_.transpose();
            // Проверка на NaN/Inf
            if (!safe_cov.allFinite()) {
                std::cout << "WARNING: covariance contains NaN/Inf, returning identity" << std::endl;
                safe_cov = Eigen::MatrixXd::Identity(S_.rows(), S_.rows()) * 0.1;
            }
        } catch (const std::exception& e) {
            std::cout << "ERROR in covariance(): " << e.what() << std::endl;
            safe_cov = Eigen::MatrixXd::Identity(S_.rows(), S_.rows()) * 0.1;
        }
        return safe_cov;
    }
    const Eigen::MatrixXd& covarianceSqrt() const { return S_; }
private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd S_;
};
class CSRF
{
public:
    CSRF(const size_t nx);
    CSRF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, const Eigen::MatrixXd& D, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::VectorXd &u, const Eigen::VectorXd &y);
    void step(const Eigen::VectorXd &u, const Eigen::VectorXd &y);
    //void predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u);
    //void update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd &y);
    const Eigen::VectorXd& state() const { return x_; }
    //const Eigen::MatrixXd& covariance() const { return ; }
private:
    void applySigmaUnitary(Eigen::MatrixXd &prearray);
    Eigen::VectorXd x_;
    Eigen::MatrixXd L_, G_, A_, C_, D_, SRe_, Sigma_p_;
};
}