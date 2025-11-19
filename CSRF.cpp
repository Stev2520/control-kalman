#include "kalman.hpp"
using namespace kalman;
CSRF::CSRF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)), L_(Eigen::MatrixXd::Zero(nx, 0)), G_(Eigen::MatrixXd::Zero(nx, 0)), initialized_(false) { }
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
    Sigma_ = Eigen::MatrixXd::Zero(rank, rank);
    size_t col = 0;
    for (int i = 0; i < eigenvalues.size(); ++i)
        if (eigenvalues(i) > 1e-10)
        {
            L_.col(col) = eigenvectors.col(i) * std::sqrt(eigenvalues(i));
            Sigma_(col, col) = 1;
            ++col;
        }
    for (int i = 0; i < eigenvalues.size(); ++i)
        if (eigenvalues(i) < -1e-10)
        {
            L_.col(col) = eigenvectors.col(i) * std::sqrt(eigenvalues(i));
            Sigma_(col, col) = -1;
            ++col;
        }
}

void CSRF::step(const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    const size_t nx = x_.size(), ny = y.size(), nu = u.size();
    assert(y.size() == SRe_.rows() && u.size() == D_.cols());
    Eigen::MatrixXd pre = Eigen::MatrixXd(nx + ny, nx + L_.cols());
    pre.block(0, 0, ny, ny) = SRe_;
    pre.block(ny, 0, nx, ny) = G_;
    pre.block(0, ny, ny, L_.cols()) = C_ * L_;
    pre.block(ny, ny, nx, L_.cols()) = A_ * L_;

}
void CSRF::step(const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    const size_t nx = x_.size(), ny = y.size(), nw = B.cols();
    
    if (!initialized_)
    {
        // Initial covariance computation as in paper
        Eigen::MatrixXd P0 = S_ * S_.transpose();  // P_{0|-1}
        Eigen::MatrixXd P1 = A * P0 * A.transpose() + B * Q * B.transpose(); // P_{1|0}
        
        // Initial increment and L_0 factorization
        Eigen::MatrixXd incP0 = P1 - P0;
        
        // Compute signature matrix Σ and L_0 from incP0 = L_0 Σ L_0ᵀ
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(incP0);
        Eigen::VectorXd eigenvalues = es.eigenvalues();
        Eigen::MatrixXd eigenvectors = es.eigenvectors();
        
        // Determine signature (n1 positive, n2 negative eigenvalues)
        n1_ = n2_ = 0;
        for (int i = 0; i < eigenvalues.size(); ++i)
        {
            if (eigenvalues(i) > 1e-10) ++n1_;
            else if (eigenvalues(i) < -1e-10) ++n2_;
        }
        
        // Build L_0 and Σ
        L_ = Eigen::MatrixXd(nx, n1_ + n2_);
        Sigma_ = Eigen::MatrixXd::Zero(n1_ + n2_, n1_ + n2_);
        
        int col = 0;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (std::abs(eigenvalues(i)) > 1e-10) {
                L_.col(col) = eigenvectors.col(i) * std::sqrt(std::abs(eigenvalues(i)));
                Sigma_(col, col) = (eigenvalues(i) > 0) ? 1 : -1;
                col++;
            }
        }
        
        // Initial G_0 computation
        Eigen::MatrixXd Re0 = R + C * P0 * C.transpose();
        Eigen::MatrixXd Re_sqrt = Eigen::LLT<Eigen::MatrixXd>(Re0).matrixL();
        G_ = A * P0 * C.transpose() * Re_sqrt.inverse();
        
        initialized_ = true;
    }
    
    // Build the prearray matrix from equation (16)
    Eigen::MatrixXd Re_sqrt = Eigen::LLT<Eigen::MatrixXd>(R + C * L_ * Sigma_ * L_.transpose() * C.transpose()).matrixL();
    
    Eigen::MatrixXd prearray(ny + nx, ny + L_.cols());
    prearray.setZero();
    prearray.block(0, 0, ny, ny) = Re_sqrt;
    prearray.block(0, ny, ny, L_.cols()) = C * L_;
    prearray.block(ny, 0, nx, ny) = G_;
    prearray.block(ny, ny, nx, L_.cols()) = A * L_;
    
    // Apply Σ-unitary transformation (simplified - would need skew Householder)
    // This is the complex part requiring indefinite norm transformations
    Eigen::MatrixXd Sigma_p = Eigen::MatrixXd::Identity(ny + L_.cols(), ny + L_.cols());
    Sigma_p.block(ny, ny, L_.cols(), L_.cols()) = Sigma_;
    
    // For now, using standard QR as approximation - NOT the true Chandrasekhar form
    Eigen::MatrixXd postarray = prearray; // Placeholder - needs Σ-unitary transform
    
    // Extract results  
    Eigen::MatrixXd Re_sqrt_new = postarray.block(0, 0, ny, ny);
    Eigen::MatrixXd L_new = postarray.block(ny, ny, nx, L_.cols());
    G_ = postarray.block(ny, 0, nx, ny);
    
    L_ = L_new;
    
    // State update - equation (17)
    x_ = A * x_ - G_ * Re_sqrt_new.inverse() * (C * x_ - y) + D * u;
}