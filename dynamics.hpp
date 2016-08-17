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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

using thrust::device_vector;
using thrust::host_vector;
using thrust::complex;

#include <cusolverDn.h>

#include "gutzwiller.hpp"

#ifdef CPU
typedef host_vector<complex<double>> state_type;
typedef host_vector<double> double_vector;
typedef host_vector<complex<double>> complex_vector;
typedef host_vector<int> int_vector;
#else
typedef device_vector<complex<double>> state_type;
typedef device_vector<double> double_vector;
typedef device_vector<complex<double>> complex_vector;
typedef device_vector<int> int_vector;
#endif

typedef vector<std::complex<double>> ode_state_type;

typedef function<array<double, L>(int, double)> SiteFunction;
typedef function<double(int, double)> SystemFunction;

struct dynamics {
//	dynamics(cublasHandle_t cublas_handle, cusolverDnHandle_t solver_handle,
//		SystemFunction U0, SiteFunction dU, SiteFunction J, SystemFunction mu) :
//		U0f(U0), dUf(dU), Jf(J), muf(mu), cublas_handle(cublas_handle), solver_handle(
//			solver_handle), Uh(Ndim * Ndim), Vh(Ndim * Ndim), Sh(Ndim), Ud(
//			Ndim * Ndim), Vd(Ndim * Ndim), Sd(Ndim), one(
//			make_cuDoubleComplex(1, 0)), zero(make_cuDoubleComplex(0, 0)), devInfo(
//			1) {
		dynamics(cublasHandle_t cublas_handle, cusolverDnHandle_t solver_handle,
			SystemFunction U0, SiteFunction dU, SiteFunction J, SystemFunction mu, SystemFunction U0p, SiteFunction Jp) :
			U0f(U0), dUf(dU), Jf(J), muf(mu), U0pf(U0p), Jpf(Jp), cublas_handle(cublas_handle), solver_handle(
				solver_handle), Uh(Ndim * Ndim), Vh(Ndim * Ndim), Sh(Ndim), one(
				make_cuDoubleComplex(1, 0)), zero(make_cuDoubleComplex(0, 0)) {
		cusolverDnZgesvd_bufferSize(solver_handle, Ndim, Ndim, &work_size);
//		work = device_vector<cuDoubleComplex>(work_size);
	}

	void operator()(const ode_state_type& fcon, ode_state_type& dfdt, const double t);

private:
	SystemFunction U0f, muf, U0pf;
	SiteFunction dUf, Jf, Jpf;
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t solver_handle;
	host_vector<cuDoubleComplex> Uh, Vh;
	host_vector<double> Sh;
//	device_vector<cuDoubleComplex> Ud, Vd;
//	device_vector<double> Sd;
	int work_size;
//	device_vector<cuDoubleComplex> work;
	cuDoubleComplex one, zero;
//	device_vector<int> devInfo;
};

#endif /* DYNAMICS_HPP_ */
