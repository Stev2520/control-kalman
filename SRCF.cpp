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
//        SQ = lltQ.matrixL();
        SQ = Q.llt().matrixL();
    }

    Eigen::LLT<Eigen::MatrixXd> lltR(R);
    if (lltR.info() != Eigen::Success) {
        std::cerr << "WARNING: R is not positive definite! Adding regularization." << std::endl;
        Eigen::MatrixXd R_reg = R + Eigen::MatrixXd::Identity(R.rows(), R.cols()) * 1e-8;
        SR = R_reg.llt().matrixL();
    } else {
//        SR = lltR.matrixL();
        SR = R.llt().matrixL();
    }
    std::cout << "SQ (" << SQ.rows() << "x" << SQ.cols() << "): " << SQ << "\n";
    std::cout << "SR (" << SR.rows() << "x" << SR.cols() << "): " << SR << "\n";
    std::cout << "S_: " << S_ << "\n";
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

        // Нижний блок: [0, A*S, B*SQ]
        prearray.block(ny, ny, nx, nx) = A_times_S;
        prearray.block(ny, nx + ny, nx, nw) = B_times_SQ;
    } catch (const std::exception& e) {
        std::cerr << "ERROR forming prearray: " << e.what() << std::endl;
        return;
    }

    std::cout << "Prearray:\n" << prearray << "\n";

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
    Eigen::MatrixXd R_mat;
    try {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(prearray.transpose());
        R_mat = qr.matrixQR().template triangularView<Eigen::Upper>();
        std::cout << "R_mat (" << R_mat.rows() << "x" << R_mat.cols() << "):\n" << R_mat << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR in QR decomposition: " << e.what() << std::endl;
        return;
    }

    if (!R_mat.allFinite()) {
        std::cerr << "ERROR: R_mat contains NaN/Inf after QR!" << std::endl;
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
        Eigen::MatrixXd R_top = R_mat.topLeftCorner(nx + ny, nx + ny);
        postarray = R_top.transpose();
        std::cout << "\n5. Postarray (" << postarray.rows() << "x" << postarray.cols() << "):\n" << postarray << "\n";
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

    // 4. Извлекаем блоки
    // post = [S_Re    0]
    //        [G      S_next]
    std::cout << "\n6. Extracting blocks:\n";
    Eigen::MatrixXd S_Re, G, S_next;
    try {
        // Извлекаем блоки БЕЗ умножения на -1
        Eigen::MatrixXd R_e_raw = postarray.block(0, 0, ny, ny);
        Eigen::MatrixXd G_raw = postarray.block(ny, 0, nx, ny);
        Eigen::MatrixXd S_next_raw = postarray.block(ny, ny, nx, nx);

        std::cout << "\nRaw blocks (before sign correction):\n";
        std::cout << "R_e_raw:\n" << R_e_raw << "\n";
        std::cout << "G_raw:\n" << G_raw << "\n";
        std::cout << "S_next_raw:\n" << S_next_raw << "\n";

// Копируем для корректировки знаков
        Eigen::MatrixXd R_e_corrected = R_e_raw;
        Eigen::MatrixXd G_corrected = G_raw;
        Eigen::MatrixXd S_next_corrected = S_next_raw;

// 1. Корректируем R_e (должен быть нижнетреугольный с положительной диагональю)
        for (int i = 0; i < ny; ++i) {
            double diag_val = R_e_raw(i, i);
            if (diag_val < 0) {
                // Инвертируем знак всей строки i в R_e и соответствующей строки в G
                R_e_corrected.row(i) = -R_e_raw.row(i);
                G_corrected.row(i) = -G_raw.row(i);
                std::cout << "Corrected sign for row " << i
                          << " (diag was " << diag_val << ")\n";
            }
        }

        // 2. Корректируем S_next (должен быть нижнетреугольный с положительной диагональю)
        for (int i = 0; i < nx; ++i) {
            double diag_val = S_next_raw(i, i);
            if (diag_val < 0) {
                // Инвертируем знак всей строки i в S_next
                S_next_corrected.row(i) = -S_next_raw.row(i);
                std::cout << "Corrected sign for S_next row " << i
                          << " (diag was " << diag_val << ")\n";
            }
        }

        Eigen::MatrixXd P_from_S = S_next_corrected * S_next_corrected.transpose();
        std::cout << "P from corrected S_next:\n" << P_from_S << "\n";

        // Присваиваем скорректированные блоки
        S_Re = R_e_corrected;
        G = G_corrected;
        S_next = S_next_corrected;

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
        // Простой прогноз как запасной вариант
        x_ = A * x_ + D * u;
        S_ = (A * S_ * S_.transpose() * A.transpose() + B * Q * B.transpose())
                .llt().matrixL();
        return;
    }

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
        // Запасной вариант: простой прогноз
        x_ = A * x_ + D * u;
        return;
    }

    // 6. Обновляем состояние и ковариацию
    Eigen::VectorXd x_old = x_;
    std::cout << "x_old: " << x_old << " \n";
    if (!z.allFinite()) {
        std::cerr << "WARNING: z contains NaN/Inf, using zero update" << std::endl;
        x_ = A * x_ + D * u;
    } else {
        // Ограничение слишком больших обновлений
        double z_norm = z.norm();
//        if (z_norm > 100.0) {
//            std::cerr << "WARNING: Large update in SRCF (z_norm = " << z_norm << "), limiting" << std::endl;
////            z = z / z_norm * 100.0;
//        }
        try {
            std::cout << "x_new = A*x + G*z + D*u:\n";
            std::cout << "  A*x = " << (A * x_).transpose() << "\n";
            std::cout << "  G*z = " << (G * z).transpose() << "\n";
            std::cout << "  D*u = " << (D * u).transpose() << "\n";
            x_ = A * x_ + G * z + D * u;
            std::cout << "  x_new = " << x_.transpose() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "ERROR updating state: " << e.what() << std::endl;
            x_ = A * x_ + D * u;
        }
    }
    S_ = S_next;
    // Дополнительная проверка: гарантируем нижнетреугольность
    S_ = S_.triangularView<Eigen::Lower>();

    // Проверка положительной определенности
    Eigen::MatrixXd P_test = S_ * S_.transpose();
    Eigen::LLT<Eigen::MatrixXd> llt_test(P_test);
    if (llt_test.info() != Eigen::Success) {
        std::cerr << "WARNING: P not positive definite after update, regularizing" << std::endl;
        // Регуляризация
        P_test += Eigen::MatrixXd::Identity(nx, nx) * 1e-8;
        S_ = P_test.llt().matrixL();
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Old state: " << x_old.transpose() << "\n";
    std::cout << "New state: " << x_.transpose() << "\n";
    std::cout << "True state: " << (A * Eigen::Vector2d(0,0) + B * u).transpose() << "\n";
}