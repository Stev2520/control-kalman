#include "kalman.hpp"
using namespace kalman;
SRIF::SRIF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)), T_(Eigen::MatrixXd::Identity(nx, nx)) { }
SRIF::SRIF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) : x_(x0), T_(P0.inverse().llt().matrixL().transpose()) { }
void SRIF::initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0)
{
    x_ = x0;
    T_ = P0.inverse().llt().matrixL().transpose();
}
void SRIF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    const size_t nx = x_.size(), ny = y.size(), nw = B.cols(), nu = u.size();
    assert(
        A.rows() == nx && A.cols() == nx &&
        B.rows() == nx && B.cols() == nw &&
        C.rows() == ny && C.cols() == nx &&
        D.rows() == nx && D.cols() == nu &&
        Q.rows() == nw && Q.cols() == nw &&
        R.rows() == ny && R.cols() == ny
    );
    Eigen::MatrixXd invSR = R.inverse().llt().matrixL().transpose();
    Eigen::MatrixXd pre = Eigen::MatrixXd::Zero(nx + ny + nw, nx + nw + 1);
    pre.block(0, 0, nw, nw) = Q.inverse().llt().matrixL().transpose();
    pre.block(nw, nw, nx, nx) = T_ * A.inverse();
    pre.block(nw, 0, nx, nw) = pre.block(nw, nw, nx, nx) * B;
    pre.block(nw, nw + nx, nx, 1) = T_ * x_;
    pre.block(nw + nx, nw, ny, nx) = invSR * C;
    pre.block(nw + nx, nw + nx, ny, 1) = invSR * y;
    
}