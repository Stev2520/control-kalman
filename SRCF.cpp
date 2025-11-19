#include "kalman.hpp"
using namespace kalman;
SRCF::SRCF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)), S_(Eigen::MatrixXd::Identity(nx, nx)) { }
SRCF::SRCF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) : x_(x0), S_(Eigen::LLT<Eigen::MatrixXd>(P0).matrixL()) { }
void SRCF::initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0)
{
    x_ = x0;
    S_ = P0.llt().matrixL();
}
void SRCF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &u, const Eigen::VectorXd &y)
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
    Eigen::MatrixXd pre = Eigen::MatrixXd::Zero(nx + ny, nx + ny + nw);
    pre.block(0, 0, ny, ny) = R.llt().matrixL();
    pre.block(0, ny, ny, nx) = C * S_;
    pre.block(ny, ny, nx, nx) = A * S_;
    pre.block(ny, nx + ny, nx, nw) = B * Q.llt().matrixL();
    const Eigen::MatrixXd post = Eigen::HouseholderQR<Eigen::MatrixXd>(pre.transpose()).matrixQR().topRows(nx + ny).triangularView<Eigen::Upper>().transpose();
    x_ = A * x_ + post.block(ny, 0, nx, ny) * post.block(0, 0, ny, ny).inverse() * (y - C * x_) + D * u;
    S_ = post.block(ny, ny, nx, nx);
}
/*void SRCF::predict(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &D, const Eigen::MatrixXd &Q, const Eigen::VectorXd &u)
{
        const int nx = x_.size(), nw = B.cols();
        assert(
            A.rows() == nx && A.cols() == nx &&
            B.rows() == nx && B.cols() == nw &&
            Q.rows() == nw && Q.cols() == nw
        );
        // Build compound matrix for QR decomposition
        Eigen::MatrixXd compound(nx + nw, nx);
        compound.setZero();
        compound.block(0, 0, nx, nx) = A * S_;
        compound.block(nx, 0, nw, nx) = Eigen::LLT<Eigen::MatrixXd>(Q).matrixL().transpose(); // For orthogonality
        S_ = Eigen::HouseholderQR<Eigen::MatrixXd>(compound).matrixQR().triangularView<Eigen::Upper>().block(0, 0, nx, nx);
        x_ = A * x_ + D * u;
}
void SRCF::update(const Eigen::MatrixXd &C, const Eigen::MatrixXd &R, const Eigen::VectorXd& y) 
{
    const int nx = x_.size(), ny = y.size();
    assert(
        C.rows() == ny && C.cols() == nx &&
        R.rows() == ny && R.cols() == ny
    );
    Eigen::MatrixXd SR = Eigen::LLT<Eigen::MatrixXd>(R).matrixL(), compound(ny + nx, nx + ny);
    compound.setZero();
    compound.block(0, 0, ny, nx) = C * S_;
    compound.block(0, nx, ny, ny) = SR;
    compound.block(ny, 0, nx, nx) = S_;
    Eigen::MatrixXd R_mat = Eigen::HouseholderQR<Eigen::MatrixXd>(compound).matrixQR().triangularView<Eigen::Upper>();
    S_ = R_mat.block(ny, 0, nx, nx);
    x_ += R_mat.block(0, 0, ny, nx).transpose() * R_mat.block(0, nx, ny, ny).inverse().transpose() * (y - C * x_);
}*/