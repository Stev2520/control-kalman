#pragma once
#include <Eigen/Dense>
#include <functional>
namespace kalman
{
class CKF
{
public:
    CKF(const size_t nx);
    CKF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y);
    //void predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u);
    //void update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd &y);
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
    //void predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u);
    //void update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd &y);
    const Eigen::VectorXd& state() const { return x_; }
    const Eigen::MatrixXd& covariance() const { return S_ * S_.transpose(); }
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
    const Eigen::MatrixXd& covariance() const { return ; }
private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd L_, G_, A_, C_, D_, SRe_, Sigma_;
};
class SRIF
{
public:
    SRIF(const size_t nx);
    SRIF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u);
    void update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd &y);
    const Eigen::VectorXd& state() const { return x_; }
    const Eigen::MatrixXd& covariance() const { return P_; }
private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    size_t nx_;
    
};
}