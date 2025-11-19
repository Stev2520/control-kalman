#include "kalman.hpp"
#include <cmath>
using namespace kalman;
CSRF::CSRF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)), L_(Eigen::MatrixXd::Zero(nx, 0)), G_(Eigen::MatrixXd::Zero(nx, 0)) { }
CSRF::CSRF(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, const Eigen::MatrixXd& D, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::VectorXd &u, const Eigen::VectorXd &y)
    : x_(x0), A_(A), C_(C), D_(D)
{
    const size_t nx = x_.size(), ny = y.size(), nw = B.cols(), nu = D.rows();
    assert(
        A.rows() == nx && A.cols() == nx &&
        B.rows() == nx && B.cols() == nw &&
        C.rows() == ny && C.cols() == nx &&
        Q.rows() == nw && Q.cols() == nw &&
        R.rows() == ny && R.cols() == ny
    );

        const Eigen::MatrixXd Re = R + C * P0 + C.transpose();
        SRe_ = Re.llt().matrixL();
        G_ = A * P0 * C.transpose() * SRe_.inverse();
        //const Eigen::MatrixXd dP = (A - A * P0 * C.transpose() * Re.inverse() * C) * P0 * A.transpose() + B * Q * B.transpose() - P0;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es((A - A * P0 * C.transpose() * Re.inverse() * C) * P0 * A.transpose() + B * Q * B.transpose() - P0);









        Eigen::MatrixXd S = P0.llt().matrixL();
        Eigen::MatrixXd pre = Eigen::MatrixXd::Zero(nx + ny, nx + ny + nw);
        pre.block(0, 0, ny, ny) = R.llt().matrixL();
        pre.block(0, ny, ny, nx) = C * S;
        pre.block(ny, ny, nx, nx) = A * S;
        pre.block(ny, nx + ny, nx, nw) = B * Q.llt().matrixL();
        const Eigen::MatrixXd post = Eigen::HouseholderQR<Eigen::MatrixXd>(pre.transpose()).matrixQR().topRows(nx + ny).triangularView<Eigen::Upper>().transpose();
        G_ = post.block(ny, 0, nx, ny);
        SRe_ = post.block(0, 0, ny, ny);
        //const Eigen::MatrixXd dP = post.block(ny, ny, nx, nx) * post.block(ny, ny, nx, nx).transpose() - P0;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(post.block(ny, ny, nx, nx) * post.block(ny, ny, nx, nx).transpose() - P0);
        
    
    
    
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    size_t n1 = 0, n2 = 0;
    for (int i = 0; i < eigenvalues.size(); ++i)
    {
        if (eigenvalues(i) > 1e-10) ++n1;
        else if (eigenvalues(i) < -1e-10) ++n2;
    }
    const size_t rank = n1 + n2;
    L_ = Eigen::MatrixXd(nx, rank);
    Sigma_p_ = Eigen::MatrixXd::Identity(ny + rank, ny + rank);
    size_t col = ny;
    for (int i = 0; i < eigenvalues.size(); ++i)
        if (eigenvalues(i) > 1e-10)
        {
            L_.col(col) = eigenvectors.col(i) * std::sqrt(eigenvalues(i));
            Sigma_p_(col, col) = 1;
            ++col;
        }
    for (int i = 0; i < eigenvalues.size(); ++i)
        if (eigenvalues(i) < -1e-10)
        {
            L_.col(col) = eigenvectors.col(i) * std::sqrt(eigenvalues(i));
            Sigma_p_(col, col) = -1;
            ++col;
        }
}
void CSRF::step(const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    const size_t nx = x_.size(), ny = y.size(), nu = u.size(), nL = L_.cols();
    assert(y.size() == SRe_.rows() && u.size() == D_.cols());
    Eigen::MatrixXd pre = Eigen::MatrixXd(nx + ny, nx + nL);
    pre.block(0, 0, ny, ny) = SRe_;
    pre.block(ny, 0, nx, ny) = G_;
    pre.block(0, ny, ny, nL) = C_ * L_;
    pre.block(ny, ny, nx, nL) = A_ * L_;
    applySigmaUnitary(pre);
    SRe_ = pre.block(0, 0, ny, ny);
    G_ = pre.block(ny, 0, nx, ny);
    L_ = pre.block(ny, ny, nx, nL);
    x_ = A_ * x_ + G_ * SRe_ * (y - C_ * x_) + D_ * u;
}
void CSRF::applySigmaUnitary(Eigen::MatrixXd& prearray)
{
    const size_t n = prearray.cols();
    
    for (size_t k = 0; k < n - 1; ++k)
    {
        Eigen::VectorXd v = prearray.block(k, k, prearray.rows() - k, 1);
        double sigma_norm = std::sqrt(std::abs(v.transpose() * Sigma_p_.block(k, k, v.size(), v.size()) * v));
        if (sigma_norm < 1e-10) continue;
        v(0) += (v(0) >= 0) ? sigma_norm : -sigma_norm;
        const double tau = (v.transpose() * Sigma_p_.block(k, k, v.size(), v.size()) * v)(0) * .5;
        if (std::abs(tau) < 1e-10) continue;
        // Apply the skew Householder transformation: I - (u * uᵀ * Σ) / τ
        for (size_t j = k; j < n; ++j)
            prearray.block(k, j, prearray.rows() - k, 1) -= v * (v.transpose() * Sigma_p_.block(k, k, v.size(), v.size()) * prearray.block(k, j, prearray.rows() - k, 1))(0) / tau;
    }
}