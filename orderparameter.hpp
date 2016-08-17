/* 
 * File:   orderparameter.hpp
 * Author: Abuenameh
 *
 * Created on November 4, 2015, 3:31 AM
 */

#ifndef ORDERPARAMETER_HPP_
#define	ORDERPARAMETER_HPP_

#include <vector>

using std::vector;

#include <thrust/complex.h>

using thrust::complex;

#include "gutzwiller.hpp"

complex<double> b0(vector<vector<complex<double>>>& f, int i);
complex<double> b1(vector<vector<complex<double>>>& f, int i, vector<double>& J, double U);
complex<double> b2(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U);
complex<double> b3(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U);
complex<double> b(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U);

#endif	/* ORDERPARAMETER_HPP_ */

