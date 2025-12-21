#include "kalman.hpp"
#include <iostream>
using namespace kalman;
CKF::CKF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)), P_(Eigen::MatrixXd::Identity(nx, nx)) { }
CKF::CKF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) : x_(x0), P_(P0) { }
void CKF::initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0)
{
    x_ = x0;
    P_ = P0;
}
void CKF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    static bool firstRun = true;
    const size_t nx = x_.size(), ny = y.size(), nw = B.cols(), nu = u.size();
    if (firstRun)
    {
        std::cout << "nx: " << nx << ", ny: " << ny << ", nw: " << nw << ", nu: " << nu << std::endl;
        std::cout << A.rows() << ' ' << A.cols() << '\n'
                  << B.rows() << ' ' << B.cols() << '\n'
                  << C.rows() << ' ' << C.cols() << '\n'
                  << D.rows() << ' ' << D.cols() << '\n'
                  << Q.rows() << ' ' << Q.cols() << '\n'
                  << R.rows() << ' ' << R.cols() << std::endl;
        firstRun = false;
    }
    assert(
        A.rows() == nx && A.cols() == nx &&
        B.rows() == nx && B.cols() == nw &&
        C.rows() == ny && C.cols() == nx &&
        D.rows() == nx && D.cols() == nu &&
        Q.rows() == nw && Q.cols() == nw &&
        R.rows() == ny && R.cols() == ny
    );
    const Eigen::MatrixXd K = A * P_ * C.transpose() * (R + C * P_ * C.transpose()).inverse();
    x_ = A * x_ + K * (y - C * x_) + D * u;
    P_ = (A - K * C) * P_ * A.transpose() + B * Q * B.transpose();
}
/*void CKF::predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u)
{
    assert(
        A.rows() == x_.size() && A.cols() == x_.size() &&
        B.rows() == x_.size() && B.cols() == Q.rows()  &&
        Q.rows() == Q.cols()                           &&
        D.rows() == x_.size() && D.cols() == u.size()
    );
    x_ = A * x_ + D * u;
    P_ = A * P_ * A.transpose() + B * Q * B.transpose();
}
void CKF::update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd& y) 
{
    assert(
        C.rows() == y.size() && C.cols() == x_.size() &&
        R.rows() == y.size() && R.cols() == y.size()
    );
    //Eigen::MatrixXd Re = R + C * P_ * C.transpose();
    Eigen::MatrixXd K = P_ * C.transpose() * (R + C * P_ * C.transpose()).inverse();
    //Eigen::VectorXd v = y - C * x_;
    x_ += K * (y - C * x_);
    P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * C) * P_;
}*/
