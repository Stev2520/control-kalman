#include <iostream>
#include "kalman.hpp"

using namespace kalman;

SRCF::SRCF(const size_t nx) : x_(Eigen::VectorXd::Zero(nx)),
                              S_(Eigen::MatrixXd::Identity(nx, nx)) { }

SRCF::SRCF(const Eigen::VectorXd &x0,
           const Eigen::MatrixXd &P0) : x_(x0),
                                        S_(Eigen::LLT<Eigen::MatrixXd>(P0).matrixL()) { }

void SRCF::initialize(const Eigen::VectorXd &x0,
                      const Eigen::MatrixXd &P0)
{
    x_ = x0;
    S_ = P0.llt().matrixL();
}

void SRCF::step(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
                const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                const Eigen::VectorXd &u, const Eigen::VectorXd &y)
{
    std::cout << "\n=== Manual SRCF Step ===\n";
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
    std::cout << "nx=" << nx << ", ny=" << ny << ", nw=" << nw  << ", nu=" << nu << "\n";

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

    // 1. Квадратные корни
    std::cout << "\n1. Cholesky decompositions:\n";
    Eigen::MatrixXd SQ, SR;
    Eigen::LLT<Eigen::MatrixXd> lltQ(Q);
    if (lltQ.info() != Eigen::Success) {
        std::cerr << "WARNING: Q is not positive definite! Adding regularization." << std::endl;
        Eigen::MatrixXd Q_reg = Q + Eigen::MatrixXd::Identity(Q.rows(), Q.cols()) * 1e-8;
        SQ = Q_reg.llt().matrixL();
    } else {
        SQ = Q.llt().matrixL();
    }

    Eigen::LLT<Eigen::MatrixXd> lltR(R);
    if (lltR.info() != Eigen::Success) {
        std::cerr << "WARNING: R is not positive definite! Adding regularization." << std::endl;
        Eigen::MatrixXd R_reg = R + Eigen::MatrixXd::Identity(R.rows(), R.cols()) * 1e-8;
        SR = R_reg.llt().matrixL();
    } else {
        SR = R.llt().matrixL();
    }
    std::cout << "SQ (" << SQ.rows() << "x" << SQ.cols() << "): " << SQ << "\n";
    std::cout << "SR (" << SR.rows() << "x" << SR.cols() << "): " << SR << "\n";
    std::cout << "Current S_: " << S_ << "\n";
    Eigen::MatrixXd P_old = S_ * S_.transpose();
    std::cout << "Initial covariance P_: " << P_old << "\n";
    std::cout << "\n1. Covariance prediction:\n";
    std::cout << "P_old Norm = " << P_old.norm() << "\n";
    if (!S_.allFinite()) {
        std::cout << "WARNING: S_ contains NaN/Inf, resetting to identity" << std::endl;
        S_ = Eigen::MatrixXd::Identity(nx, nx) * 0.1;
    }

    // 2. Формирование пре-массива
    std::cout << "\n2. Computing prearray blocks:\n";
    Eigen::MatrixXd C_times_S = C * S_;
    Eigen::MatrixXd A_times_S = A * S_;
    Eigen::MatrixXd B_times_SQ = B * SQ;

    std::cout << "C*S (" << C_times_S.rows() << "x" << C_times_S.cols() << "):\n" << C_times_S << "\n";
    std::cout << "A*S (" << A_times_S.rows() << "x" << A_times_S.cols() << "):\n" << A_times_S << "\n";
    std::cout << "B*SQ (" << B_times_SQ.rows() << "x" << B_times_SQ.cols() << "):\n" << B_times_SQ << "\n";

    std::cout << "\n3. Building prearray (" << (nx + ny) << "x" << (nx + ny + nw) << "):\n";
    Eigen::MatrixXd prearray = Eigen::MatrixXd::Zero(nx + ny, nx + ny + nw);

    try {
        // Верхний блок: [SR, C*S, 0]
        prearray.block(0, 0, ny, ny) = SR;
        prearray.block(0, ny, ny, nx) = C_times_S;
        prearray.block(0, nx + ny, ny, nw).setZero();;

        // Нижний блок: [0, A*S, B*SQ]
        prearray.block(ny, 0, nx, ny).setZero();
        prearray.block(ny, ny, nx, nx) = A_times_S;
        prearray.block(ny, nx + ny, nx, nw) = B_times_SQ;
    } catch (const std::exception& e) {
        std::cerr << "ERROR forming prearray: " << e.what() << std::endl;
        return;
    }

    std::cout << "Prearray (" << prearray.rows() << "x" << prearray.cols() << "):\n"
              << prearray << "\n";

    if (!prearray.allFinite()) {
        std::cout << "ERROR: M contains NaN/Inf!" << std::endl;
        std::cout << "SR norm: " << SR.norm() << std::endl;
        std::cout << "C*S_ norm: " << (C*S_).norm() << std::endl;
        std::cout << "A*S_ norm: " << (A*S_).norm() << std::endl;
        std::cout << "B*SQ norm: " << (B*SQ).norm() << std::endl;
        return;
    }

    // 3. QR разложение
    std::cout << "\n4. QR decomposition:\n";
    Eigen::MatrixXd R_mat, Q_mat;
    try {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(prearray.transpose());
        R_mat = qr.matrixQR().template triangularView<Eigen::Upper>();
        std::cout << "R_mat (" << R_mat.rows() << "x" << R_mat.cols() << "):\n" << R_mat << "\n";
        Q_mat = qr.matrixQR().template triangularView<Eigen::Lower>();
        std::cout << "Q_mat (" << Q_mat.rows() << "x" << Q_mat.cols() << "):\n" << Q_mat << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR in QR decomposition: " << e.what() << std::endl;
        return;
    }

    if (!R_mat.allFinite()) {
        std::cerr << "ERROR: R_mat contains NaN/Inf after QR!" << std::endl;
        return;
    }

    if (!Q_mat.allFinite()) {
        std::cerr << "ERROR: Q_mat contains NaN/Inf after QR!" << std::endl;
        return;
    }

    // Берем первые (p+n) строк
    if (R_mat.rows() < ny + nx || R_mat.cols() < ny + nx) {
        std::cerr << "ERROR: R_mat too small: " << R_mat.rows() << "x" << R_mat.cols()
                  << " (need at least " << ny + nx << "x" << ny + nx << ")" << std::endl;
        return;
    }

    Eigen::MatrixXd postarray;
    // Берем нужный блок: первые (nx+ny) строк и столбцов
    try {
        Eigen::MatrixXd R_top = -1 * R_mat.topLeftCorner(nx + ny, nx + ny);
        postarray = R_top.transpose();
        std::cout << "\nPostarray (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR extracting postarray: " << e.what() << std::endl;
        return;
    }

    if (!postarray.allFinite()) {
        std::cout << "ERROR: postarray contains NaN/Inf!" << std::endl;
        return;
    }

    if (postarray.rows() < ny + nx || postarray.cols() < ny + nx) {
        std::cout << "ERROR: postarray too small: "
                  << postarray.rows() << "x" << postarray.cols()
                  << ", expected at least " << ny + nx << "x" << ny + nx << std::endl;
        return;
    }

    for (int i = 0; i < ny; ++i) {
        if (postarray(i, i) < 0) {
            postarray.row(i) = -postarray.row(i);
        }
    }

    for (int i = 0; i < nx; ++i) {
        int row_idx = ny + i;
        if (postarray(row_idx, row_idx) < 0) {
            postarray.row(row_idx) = -postarray.row(row_idx);
        }
    }
    std::cout << "\nPostarray after update (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";

    // 4. Извлекаем блоки
    // post = [S_Re    0]
    //        [G      S_next]
    std::cout << "\n6. Extracting blocks:\n";
    Eigen::MatrixXd S_Re, G, S_next;
    try {
        S_Re = postarray.block(0, 0, ny, ny);
        G = postarray.block(ny, 0, nx, ny);
        S_next = postarray.block(ny, ny, nx, nx);
        std::cout << "\nCorrected blocks:\n";
        std::cout << "S_Re:\n" << S_Re << "\n";
        std::cout << "G:\n" << G << "\n";
        std::cout << "S_next:\n" << S_next << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR extracting blocks: " << e.what() << std::endl;
        return;
    }

    if (!S_next.allFinite()) {
        std::cerr << "WARNING: S_next contains NaN/Inf, using fallback" << std::endl;
        return;
    }

    // === 5. ВЫЧИСЛЕНИЕ ЭКВИВАЛЕНТНЫХ ВЕЛИЧИН ===
    std::cout << "\n5. Computing equivalent quantities:\n";

    // P_next из SRCF (должно совпадать с CKF P_next)
    Eigen::MatrixXd P_next_srcf = S_next * S_next.transpose();
    std::cout << "SRCF P_next = S_next * S_next^T:\n" << P_next_srcf << "\n";

    // Калманово усиление из SRCF: K = G * inv(S_Re)
    Eigen::MatrixXd K_srcf = G * S_Re.inverse();
    std::cout << "SRCF K = G * inv(S_Re):\n" << K_srcf << "\n";

    // S матрица инноваций: S = S_Re * S_Re^T
    Eigen::MatrixXd S_srcf = S_Re * S_Re.transpose();
    std::cout << "SRCF S = S_Re * S_Re^T:\n" << S_srcf << "\n";

    std::cout << "Condition number of S_Re: "
              << S_Re.norm() * S_Re.inverse().norm() << "\n";

    // 5. Обновление состояния (численно устойчивое)
    std::cout << "\n7. State update:\n";
    Eigen::VectorXd innov = y - C * x_;
    std::cout << "Innovation: " << innov.transpose() << "\n";

    // Проверка обусловленности S_Re
    if (S_Re.diagonal().cwiseAbs().minCoeff() < 1e-12) {
        std::cerr << "WARNING: S_Re has very small diagonal elements, adding regularization" << std::endl;
        S_Re.diagonal().array() += 1e-8;
    }

    // Решаем S_Re * z = innov (S_Re - нижнетреугольная)
    Eigen::VectorXd z;
    try {
        z = S_Re.triangularView<Eigen::Lower>().solve(innov);
        std::cout << "z = S_Re\\innov: " << z.transpose() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR solving triangular system: " << e.what() << std::endl;
        return;
    }

    // 6. Обновляем состояние и ковариацию
    Eigen::VectorXd x_old = x_;
    Eigen::VectorXd x_pred = A * x_ + D * u;
    Eigen::VectorXd x_new = x_pred + G * z;
    std::cout << "x_old: " << x_old << " \n";
    std::cout << "x_pred = A*x + D*u: " << x_pred.transpose() << "\n";
    std::cout << "x_new = x_pred + G*z: " << x_new.transpose() << "\n";

    if (!z.allFinite()) {
        std::cerr << "WARNING: z contains NaN/Inf, using zero update" << std::endl;
        x_ = x_pred;
    } else {
        // Ограничение слишком больших обновлений
        try {
            std::cout << "x_new = A*x + G*z + D*u:\n";
            std::cout << "  A*x = " << (A * x_).transpose() << "\n";
            std::cout << "  G*z = " << (G * z).transpose() << "\n";
            std::cout << "  D*u = " << (D * u).transpose() << "\n";
            x_ = x_new;
            std::cout << "  x_new = " << x_.transpose() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "ERROR updating state: " << e.what() << std::endl;
        }
    }
    std::cout << "S_Re (should be Cholesky factor of S):\n" << S_Re << "\n";
    std::cout << "S_next (Cholesky factor of P_pred):\n" << S_next << "\n";
    // === 8. СРАВНЕНИЕ С CKF ===
    std::cout << "\n=== FOR CKF COMPARISON (SRCF Summary) ===\n";
    std::cout << "P_old (should match CKF P_old):\n" << P_old << "\n";
    std::cout << "K (should match CKF K = G*inv(S_Re)):\n" << G << "\n";
    std::cout << "Old x state: " << x_old.transpose() << "\n";
    std::cout << "Final new x state: " << x_.transpose() << "\n";
    std::cout << "Final covariance P_new (should match SKF P_new):\n" << P_next_srcf << "\n";
    std::cout << "True state: " << (A * Eigen::Vector2d(0,0) + D * u).transpose() << "\n";

    S_ = S_next;
    // Дополнительная проверка: гарантируем нижнетреугольность
    S_ = S_.triangularView<Eigen::Lower>();

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_test(P_next_srcf);
    if (llt_test.info() != Eigen::Success) {
        std::cerr << "WARNING: P not positive definite after update, regularizing" << std::endl;
        // Регуляризация
        P_next_srcf += Eigen::MatrixXd::Identity(nx, nx) * 1e-8;
        S_ = P_next_srcf.llt().matrixL();
    }

    // === ПРОВЕРКА ===
    std::cout << "\n=== CHECKS ===\n";

    // Проверка положительной определенности
    Eigen::LLT<Eigen::MatrixXd> llt_pred_srcf(P_old);
    Eigen::LLT<Eigen::MatrixXd> llt_new_srcf(P_next_srcf);

    std::cout << "SRCF P_pred positive definite: "
              << (llt_pred_srcf.info() == Eigen::Success ? "YES" : "NO") << "\n";
    std::cout << "SRCF P_new positive definite: "
              << (llt_new_srcf.info() == Eigen::Success ? "YES" : "NO") << "\n";

    // Проверка симметрии
    double asym_pred_srcf = (P_old - P_old.transpose()).norm() / P_old.norm();
    double asym_new_srcf = (P_next_srcf - P_next_srcf.transpose()).norm() / P_next_srcf.norm();

    std::cout << "SRCF P_pred asymmetry: " << asym_pred_srcf << "\n";
    std::cout << "SRCF P_new asymmetry: " << asym_new_srcf << "\n";
}