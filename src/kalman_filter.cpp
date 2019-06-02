#include "ros/ros.h"
#include "sensor_msgs/NavSatFix.h"
#include "sensor_msgs/FluidPressure.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "Eigen/Dense"
#include "unsupported/Eigen/src/MatrixFunctions/MatrixExponential.h"

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace Eigen;

//Constants
float g = 9.8f;        // gravity [m/s^2] (could come from a gravity model)

// Values required to calculate initial value for k in Baro observation
// equation:
float M = 28.97e-3;  // Air molar mass [kg/mol]
float R = 8.314;  // Universal gas constant [J/kg/K]
float T = 298.0;  // Air temperature (carefull about the units)

double last_stamp = 0;

// KALMAN filter uncertainty parameters
//Standard deviation of the GPS position white noise
float sigma_z_gps  = 5;       // [m]

//Standard deviation of the BARO readings white noise
float sigma_z_baro = 5;       // [Pa]

// Process model noise
float sigma_a  = 1;           // [m/s^2 /s]
float sigma_p0 = 0.01;           // [Pa /s]
float sigma_k  = 1e-9;           // [s^2/m^2 /s]

// Generate the corresponding coviariance matrices
// INITIAL VALUES
//States

Matrix<float,6,1> X_0;

//Covariances

Matrix<float,6,6> P_0;

//A PRIORI estimate
Matrix<float,6,1> x_tilde;
Matrix<float,6,6> P_tilde;


// State vector initial A POSTERIORI covariance matrix P_0
Matrix<float,6,1> x_hat;
Matrix<float,6,6> P_hat;

// Observations covariance matrices
float R_gps  = sigma_z_gps*sigma_z_gps;
float R_baro = sigma_z_baro*sigma_z_baro;

//Calculation of Kalman filter matrices
//Process model noise [PSD]

Matrix<float,6,6> Q;
Matrix<float,6,6> Q_w;

// Dynamic matrix
Matrix<float,6,6> F = Matrix<float,6,6>::Zero();

// Noise shaping matrix
Matrix<float,6,6> G = Matrix<float,6,6>::Zero();

ros::Publisher height_pub;

void init()
{
    ROS_INFO("Init");
    X_0 << 0.0f, 0.0f, 516.30f, M/(R*T), 516.30f, 96294.0f;
    P_0 << 0.5,0,0,0,0,0,
        0,0.5,0,0,0,0,
        0,0,5,0,0,0,
        0,0,0,1e-6,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0;
    x_tilde = X_0;
    P_tilde = P_0;
    x_hat = X_0;
    P_hat = P_0;
    Q << sigma_a*sigma_a,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,sigma_k*sigma_k,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,sigma_p0*sigma_p0;
    F(1,0) = 1;
    F(2,1) = 1;
    G(2,0) = 1;
    G(3,1) = 1;
    G(5,2) = 1;
}

Matrix<float,6,6> compute_Q_w(float dt)
{
    MatrixXf A(12,12);
    MatrixXf B(12,12);
    
    A << -F, G*Q*G.transpose(), Matrix<float,6,6>::Zero(), F.transpose();
    
    A = A*dt;
    B = A.exp();
    
    Matrix<float,6,6> PHI;
    PHI = B.bottomRightCorner(6,6);
    PHI.transposeInPlace();
    
    Q_w = PHI*B.topLeftCorner(6,6);
    return PHI;
}

void prediction(float dt)
{
	Matrix<float,6,6> PHI = compute_Q_w(dt);
    
    x_tilde = PHI*x_hat;
    P_tilde = PHI*P_hat*PHI.transpose() + Q_w;
}

void BARO_update_const_a(float p)
{
    float h = x_tilde(2);
    float k = x_tilde(3);
    float h0 = x_tilde(4);
    float p0 = x_tilde(5);
    cout << "x_tilde:\n" << x_tilde << endl;
    float A = exp(-k*g*(h-h0));
    //ROS_INFO("A : %f", A);
    Matrix<float,1,6> H_baro;
    H_baro << 0, 0, -k*g*p0, -g*(h-h0)*p0, k*g*p0, 1;
    H_baro *= A;
    cout << "H_baro:\n" << H_baro << endl;
    
    Matrix<float,6,1> K;
    K = P_tilde*H_baro.transpose()/(H_baro*P_tilde*H_baro.transpose() + R_baro);
    cout << "K:\n" << K << endl;
    ROS_INFO("p: %f\tpi: %f\toffset: %f", p, (float)(H_baro*x_tilde), (H_baro*x_tilde)+2*A*p0*k*g*(h-h0));
    x_hat = x_tilde + K*(p - H_baro*x_tilde - 2*A*p0*k*g*(h-h0));
    P_hat = (Matrix<float,6,6>::Identity() - K*H_baro)*P_tilde;
    cout << "x_hat:\n" << x_hat << endl;
}

void GPS_update_const_a(float z)
{
	Matrix<float,1,6> H_gps;
	H_gps << 0, 0, 1, 0, 0, 0;
	
    Matrix<float,6,1> K;
    K = P_tilde*H_gps.transpose()/(H_gps*P_tilde*H_gps.transpose() + R_gps);

	x_hat = x_tilde + K*(z - H_gps*x_tilde);
	P_hat = (Matrix<float,6,6>::Identity() - K*H_gps)*P_tilde;
}

void pos_cb(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    double stamp = msg->header.stamp.toSec();
    double dt = stamp - last_stamp;
    if(dt > 0)
    {
        //ROS_INFO("GPS update");
        prediction(dt);
        GPS_update_const_a(msg->altitude);
        last_stamp = stamp;

        geometry_msgs::Vector3Stamped sent_msg;
        sent_msg.header = msg->header;
        sent_msg.vector.z = x_hat(2);

        height_pub.publish(sent_msg);
    }
}

void pressure_cb(const sensor_msgs::FluidPressure::ConstPtr& msg)
{
    double stamp = msg->header.stamp.toSec();
    double dt = stamp - last_stamp;
    if(dt > 0)
    {
        //ROS_INFO("Baro update");
        prediction(dt);
        BARO_update_const_a(msg->fluid_pressure/100);
        last_stamp = stamp;

        geometry_msgs::Vector3Stamped sent_msg;
        sent_msg.header = msg->header;
        sent_msg.vector.z = x_hat(2);

        height_pub.publish(sent_msg);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "kalman_filter");

    ros::NodeHandle n;

    ros::Subscriber pos_sub = n.subscribe("/iris_1/global_position/raw/fix", 1000, pos_cb);

    ros::Subscriber pressure_sub = n.subscribe("/iris_1/imu/atm_pressure", 1000, pressure_cb);

    height_pub = n.advertise<geometry_msgs::Vector3Stamped>("/g18/height", 1000);

    init();

    ros::spin();

    return 0;
}
