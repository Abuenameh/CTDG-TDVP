/*
 * groundstate.hpp
 *
 *  Created on: Aug 9, 2016
 *      Author: Abuenameh
 */

#ifndef GROUNDSTATE_HPP_
#define GROUNDSTATE_HPP_

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/complex.h>

//using thrust::host_vector;
//using thrust::device_vector;
//using thrust::complex;
//using thrust::iterator_adaptor;
//using thrust::use_default;
//using thrust::counting_iterator;

//#include <cppoptlib/meta.h>
//#include <cppoptlib/problem.h>
//
//using cppoptlib::Problem;

#include <vector>

using std::vector;

#include <dlib/optimization.h>

using dlib::matrix;

typedef matrix<double,0,1> column_vector;

#include "gutzwiller.hpp"

//#ifdef CPU
//typedef host_vector<complex<double>> state_type;
//typedef host_vector<double> double_vector;
//typedef host_vector<complex<double>> complex_vector;
//typedef host_vector<int> int_vector;
//#else
//typedef device_vector<complex<double>> state_type;
//typedef device_vector<double> double_vector;
//typedef device_vector<complex<double>> complex_vector;
//typedef device_vector<int> int_vector;
//#endif
//
//template<typename T>
//using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

class energy {
public:
	typedef ::column_vector column_vector;
	typedef matrix<double> general_matrix;

	energy(double U0, vector<double> dU, vector<double> J, double mu) : U0(U0), dU(dU), J(J), mu(mu) {}

	double value(const vector<double>& x);

	void gradient(const vector<double> &x, vector<double> &grad);

	/*
	double operator()(const column_vector& x) const;

	void get_derivative_and_hessian(const column_vector& x,
		column_vector& grad, matrix<double>& hess) const;
		*/

private:
	double U0;
	vector<double> dU;
	vector<double> J;
	double mu;
};



#endif /* GROUNDSTATE_HPP_ */
