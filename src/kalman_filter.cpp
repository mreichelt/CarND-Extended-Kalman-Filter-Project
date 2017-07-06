#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - H_ * x_;

    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    //new estimate
    x_ = x_ + K * y;
    long size = x_.size();
    MatrixXd I = MatrixXd::Identity(size, size);
    P_ = (I - K * H_) * P_;
}

double KalmanFilter::normalizeAngle(double angle) {
    double TWO_PI = 2.0 * M_PI;
    return angle - TWO_PI * floor((angle + M_PI) / TWO_PI);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // convert cartesian to polar coordinates
    double
            px = x_(0),
            py = x_(1),
            vx = x_(2),
            vy = x_(3);
    double ro = sqrt(px * px + py * py),
            theta = normalizeAngle(atan2(py, px)),
            ro_dot = (px * vx + py * vy) / ro;

    VectorXd h(3);
    h << ro, theta, ro_dot;

    VectorXd y = z - h;

    // now continue using the same ol' kalman filter equations
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    //new estimate
    x_ = x_ + (K * y);
    long size = x_.size();
    MatrixXd I = MatrixXd::Identity(size, size);
    P_ = (I - K * H_) * P_;
}
