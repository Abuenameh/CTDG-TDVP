/*
 * dynamics.hpp
 *
 *  Created on: Aug 5, 2016
 *      Author: Abuenameh
 */

#ifndef DYNAMICS_HPP_
#define DYNAMICS_HPP_

#include <array>
#include <functional>
#include <complex>
#include <vector>

using std::array;
using std::vector;
using std::function;

//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/complex.h>
//
//using thrust::device_vector;
//using thrust::host_vector;
//using thrust::complex;

//#include <Eigen/Sparse>
//#include <Eigen/SparseQR>
//
//using Eigen::SparseMatrix;
//using Eigen::SparseQR;
//using Eigen::COLAMDOrdering;

//#include <cusolverDn.h>

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

typedef vector<std::complex<double>> ode_state_type;

typedef function<array<double, L>(int, double)> SiteFunction;
typedef function<double(int, double)> SystemFunction;

struct dynamics {
	dynamics(SystemFunction U0, SiteFunction dU, SiteFunction J,
			SystemFunction mu, SystemFunction U0p, SiteFunction Jp) :
			U0f(U0), dUf(dU), Jf(J), muf(mu), U0pf(U0p), Jpf(Jp) {
	}

	void operator()(const ode_state_type& fcon, ode_state_type& dfdt,
			const double t);

private:
	SystemFunction U0f, muf, U0pf;
	SiteFunction dUf, Jf, Jpf;
};

#endif /* DYNAMICS_HPP_ */
