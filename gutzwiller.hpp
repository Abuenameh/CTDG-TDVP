/*
 * gutzwiller.hpp
 *
 *  Created on: Aug 5, 2016
 *      Author: Abuenameh
 */

#ifndef GUTZWILLER_HPP_
#define GUTZWILLER_HPP_

#include <cmath>

const int L = 25;
const int nmax = 7;
const int N = 1;

const int Ndim = N * L * (nmax + 1);

inline int mod(int i) {
	return (i + L) % L;
}

inline int in(int i, int n) {
	return i * (nmax + 1) + n;
}

inline int in(int i, int n, int j, int m) {
	return in(i, n) * L*(nmax+1) + in(j,m);
}

inline int in(int j, int i, int n) {
	return j * L * (nmax + 1) + i * (nmax + 1) + n;
}

inline int ij(int i, int j) {
	return i * Ndim + j;
}

inline double g(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}




#endif /* GUTZWILLER_HPP_ */
