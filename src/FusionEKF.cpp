#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    Hj_ << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;



    // initialize kalman filter variables

    // will be overridden by first measurement
    VectorXd x(4);

    MatrixXd P(4, 4);
    P << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    MatrixXd F(4, 4);
    F << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

    // will be set by each measurement
    MatrixXd Q(4, 4);

    ekf_.Init(x, P, F, H_laser_, R_laser_, Q);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);

        //set the state with the initial location and zero velocity
        VectorXd raw = measurement_pack.raw_measurements_;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            double ro = raw[0],
                    theta = raw[1],
                    ro_dot = raw[2];

            // convert polar coordinates from radar to cartesian
            double px = ro * cos(theta),
                    py = ro * sin(theta),
                    vx = ro_dot * cos(theta),
                    vy = ro_dot * sin(theta);

            ekf_.x_ << px, py, vx, vy;

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            double px = raw[0],
                    py = raw[1];
            // laser has no velocity information, so vx = vy = 0
            ekf_.x_ << px, py, 0, 0;
        }

        // initialize time
        previous_timestamp_ = measurement_pack.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    // compute time delta in seconds
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    // update F
    ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

    // update Q noise covariance
    double noise = 9;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    ekf_.Q_ << dt4 / 4 * noise, 0, dt3 / 2 * noise, 0,
            0, dt4 / 4 * noise, 0, dt3 / 2 * noise,
            dt3 / 2 * noise, 0, dt2 * noise, 0,
            0, dt3 / 2 * noise, 0, dt2 * noise;

    ekf_.Predict();



    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    VectorXd raw = measurement_pack.raw_measurements_;

    // TODO: update state & covariance matrices
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.UpdateEKF(raw);
    } else {
        // Laser updates
        ekf_.Update(raw);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
