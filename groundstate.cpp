/*
 * groundstate.cpp
 *
 *  Created on: Aug 8, 2016
 *      Author: Abuenameh
 */

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;
using std::ostream_iterator;

#include <thrust/functional.h>
#include <thrust/tabulate.h>

using thrust::equal_to;
using thrust::multiplies;
using thrust::tabulate;
using thrust::iterator_core_access;

#include <nlopt.hpp>

#include "gutzwiller.hpp"
#include "groundstate.hpp"
//#include "mathematica.hpp"

extern void hamiltonian(state_type& fc, state_type& f, const double_vector& U0,
	const double_vector& dU, const double_vector& J, const double_vector& mu,
	complex_vector& norm1, complex_vector& norm2, complex_vector& norm3, const double_vector U0p, const double_vector& Jp,
	state_type& H);

template<typename Iterator>
class repeat_iterator: public iterator_adaptor<repeat_iterator<Iterator>,
	Iterator, use_default, use_default, use_default, use_default> {
public:
	typedef iterator_adaptor<repeat_iterator<Iterator>, Iterator, use_default,
		use_default, use_default, use_default> super_t;

	__host__ __device__
	repeat_iterator(const Iterator &x, int n) :
		super_t(x), begin(x), n(n) {
	}

	friend class iterator_core_access;

private:
	unsigned int n;

	const Iterator begin;

	__host__ __device__
	typename super_t::reference dereference() const {
		return *(begin + (this->base() - begin) / n);
	}

};

template<typename T>
__host__ __device__
repeat_iterator<T> make_repeat_iterator(T x, int n) {
	repeat_iterator<T> a(x, n);
	return a;
}

template<typename T>
struct square {
	__host__ __device__
	T operator()(const T& x) const {
		return x * x;
	}
};

template<class T>
struct divideby {

	divideby(T d) :
		d(d) {
	}

	__host__ __device__
	T operator()(int i) {
		return i / d;
	}

	T d;
};

template<typename T>
struct diff {

	diff(int n) :
		n(n) {
	}

	__host__ __device__
	T operator()(int m) {
		if (n == m) {
			return 1;
		} else {
			return 0;
		}
	}

	int n;
};

template<typename T>
struct normop {
	__host__ __device__
	T operator()(const complex<T>& x) const {
		return norm(x);
	}
};

template<typename T>
struct conjop {
	__host__ __device__
	complex<T> operator()(const complex<T>& x) const {
		return conj(x);
	}
};

double energy::operator()(const column_vector& x) const {
	host_vector<complex<double>> fh(L * (nmax + 1)), fch(L * (nmax + 1));
	for (int i = 0; i < L * (nmax + 1); i++) {
		fh[i] = complex<double>(x(2 * i), x(2 * i + 1));
		fch[i] = complex<double>(x(2 * i), -x(2 * i + 1));
	}

	state_type f = fh, fc = fch;

	int_vector keys(L * (nmax + 1)), okeys(L);

	auto nmaxrep = make_repeat_iterator(counting_iterator<int>(0), nmax + 1);
	copy(nmaxrep, nmaxrep + L * (nmax + 1), keys.begin());

	complex_vector norms(L * (nmax + 1));
	transform(fc.begin(), fc.end(), f.begin(), norms.begin(),
		multiplies<complex<double>>());

	complex_vector normi(L);
	reduce_by_key(keys.begin(), keys.end(), norms.begin(), okeys.begin(),
		normi.begin());

	auto Lrep = make_repeat_iterator(counting_iterator<int>(0), L);
	copy(Lrep, Lrep + L, keys.begin());

	complex_vector norm0(1);
	reduce_by_key(keys.begin(), keys.begin() + L, normi.begin(), okeys.begin(),
		norm0.begin(), equal_to<int>(), multiplies<complex<double>>());

	complex_vector norm1(L), norm2(L), norm3(L);
	for (int i = 0; i < L; i++) {
		norm1[i] = norm0[0] / normi[i];
		norm2[i] = norm1[i] / normi[mod(i + 1)];
		norm3[i] = norm2[i] / normi[mod(i + 2)];
	}

	state_type H(1);
	double_vector U0p(1, 0), Jp(L, 0);
	hamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);

	return H[0].real() / norm0[0].real();
}

void energy::get_derivative_and_hessian(const column_vector& x,
	column_vector& grad, matrix<double>& hess) const {
	host_vector<complex<double>> fh(L * (nmax + 1)), fch(L * (nmax + 1));
	for (int i = 0; i < L * (nmax + 1); i++) {
		fh[i] = complex<double>(x(2 * i), x(2 * i + 1));
		fch[i] = complex<double>(x(2 * i), -x(2 * i + 1));
	}

	state_type f0 = fh, fc0 = fch;

	int_vector keys(L * (nmax + 1)), okeys(L);
	int_vector nmaxkeys(L * (nmax + 1));
	int_vector Lkeys(L);

	auto nmaxrep = make_repeat_iterator(counting_iterator<int>(0), nmax + 1);
	copy(nmaxrep, nmaxrep + L * (nmax + 1), nmaxkeys.begin());

	complex_vector norms(L * (nmax + 1));
	transform(fc0.begin(), fc0.end(), f0.begin(), norms.begin(),
		multiplies<complex<double>>());

	complex_vector normi(L);
	reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), norms.begin(),
		okeys.begin(), normi.begin());

	auto Lrep = make_repeat_iterator(counting_iterator<int>(0), L);
	copy(Lrep, Lrep + L, Lkeys.begin());

	complex_vector norm0(1);
	reduce_by_key(Lkeys.begin(), Lkeys.end(), normi.begin(), okeys.begin(),
		norm0.begin(), equal_to<int>(), multiplies<complex<double>>());
	complex_vector norm00 = norm0;

	complex_vector norm1(L), norm2(L), norm3(L);
	for (int i = 0; i < L; i++) {
		norm1[i] = norm0[0] / normi[i];
		norm2[i] = norm1[i] / normi[mod(i + 1)];
		norm3[i] = norm2[i] / normi[mod(i + 2)];
	}

	state_type f(L * (nmax + 1)), fc(L * (nmax + 1));
	f = f0;
	fc = fc0;

	state_type H(1);
	double_vector U0p(1, 0), Jp(L, 0);
	hamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);
	state_type H0 = H;

	vector<complex_vector> dnorms(2 * L * (nmax + 1),
		complex_vector(L * (nmax + 1)));
	vector<complex_vector> dnormi(2 * L * (nmax + 1), complex_vector(L));
	vector<complex_vector> dnorm0(2 * L * (nmax + 1), complex_vector(1));
	vector<complex_vector> dnorm1(2 * L * (nmax + 1), complex_vector(L)),
		dnorm2(2 * L * (nmax + 1), complex_vector(L)), dnorm3(
			2 * L * (nmax + 1), complex_vector(L));

	vector<complex_vector> dcnorms(2 * L * (nmax + 1),
		complex_vector(L * (nmax + 1)));
	vector<complex_vector> dcnormi(2 * L * (nmax + 1), complex_vector(L));
	vector<complex_vector> dcnorm0(2 * L * (nmax + 1), complex_vector(1));
	vector<complex_vector> dcnorm1(2 * L * (nmax + 1), complex_vector(L)),
		dcnorm2(2 * L * (nmax + 1), complex_vector(L)), dcnorm3(
			2 * L * (nmax + 1), complex_vector(L));

	vector<complex<double>> dH(L * (nmax + 1));
	vector<complex<double>> dcH(L * (nmax + 1));

	grad = column_vector(2 * L * (nmax + 1));
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			f = f0;
			tabulate(f.begin() + i * (nmax + 1),
				f.begin() + (i + 1) * (nmax + 1), diff<double>(n));
			transform(f.begin(), f.end(), fc0.begin(), dnorms[in(i, n)].begin(),
				multiplies<complex<double>>());
			reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(),
				dnorms[in(i, n)].begin(), okeys.begin(),
				dnormi[in(i, n)].begin());
			reduce_by_key(Lkeys.begin(), Lkeys.end(), dnormi[in(i, n)].begin(),
				okeys.begin(), dnorm0[in(i, n)].begin(), equal_to<int>(),
				multiplies<complex<double>>());
			for (int j = 0; j < L; j++) {
				dnorm1[in(i, n)][j] = dnorm0[in(i, n)][0] / dnormi[in(i, n)][j];
				dnorm2[in(i, n)][j] = dnorm1[in(i, n)][j]
					/ dnormi[in(i, n)][mod(j + 1)];
				dnorm3[in(i, n)][j] = dnorm2[in(i, n)][j]
					/ dnormi[in(i, n)][mod(j + 2)];
			}
			hamiltonian(fc0, f, U0, dU, J, mu, dnorm1[in(i, n)],
				dnorm2[in(i, n)], dnorm3[in(i, n)], U0p, Jp, H);
			dH[in(i, n)] = H[0];
			fc = fc0;
			tabulate(fc.begin() + i * (nmax + 1),
				fc.begin() + (i + 1) * (nmax + 1), diff<double>(n));
			transform(fc.begin(), fc.end(), f0.begin(),
				dcnorms[in(i, n)].begin(), multiplies<complex<double>>());
			reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(),
				dcnorms[in(i, n)].begin(), okeys.begin(),
				dcnormi[in(i, n)].begin());
			reduce_by_key(Lkeys.begin(), Lkeys.end(), dcnormi[in(i, n)].begin(),
				okeys.begin(), dcnorm0[in(i, n)].begin(), equal_to<int>(),
				multiplies<complex<double>>());
			for (int j = 0; j < L; j++) {
				dcnorm1[in(i, n)][j] = dcnorm0[in(i, n)][0]
					/ dcnormi[in(i, n)][j];
				dcnorm2[in(i, n)][j] = dcnorm1[in(i, n)][j]
					/ dcnormi[in(i, n)][mod(j + 1)];
				dcnorm3[in(i, n)][j] = dcnorm2[in(i, n)][j]
					/ dcnormi[in(i, n)][mod(j + 2)];
			}
			hamiltonian(fc, f0, U0, dU, J, mu, dcnorm1[in(i, n)],
				dcnorm2[in(i, n)], dcnorm3[in(i, n)], U0p, Jp, H);
			dcH[in(i, n)] = H[0];
			complex<double> dHN = dcH[in(i, n)] / norm00[0]
				- H0[0] * dcnorm0[in(i, n)][0] / (norm00[0] * norm00[0]);
			grad(2 * in(i, n)) = 2 * dHN.real();
			grad(2 * in(i, n) + 1) = 2 * dHN.imag();
		}
	}

	vector<complex_vector> ddnorms(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L * (nmax + 1)));
	vector<complex_vector> ddnormi(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L));
	vector<complex_vector> ddnorm0(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(1));
	vector<complex_vector> ddnorm1(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L)), ddnorm2(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L)), ddnorm3(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L));

	vector<complex_vector> ddcnorms(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L * (nmax + 1)));
	vector<complex_vector> ddcnormi(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L));
	vector<complex_vector> ddcnorm0(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(1));
	vector<complex_vector> ddcnorm1(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L)), ddcnorm2(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L)), ddcnorm3(2 * L * (nmax + 1) * 2 * L * (nmax + 1),
		complex_vector(L));

	vector<complex<double>> ddH(L * (nmax + 1) * L * (nmax + 1));
	vector<complex<double>> ddcH(L * (nmax + 1) * L * (nmax + 1));

	hess = matrix<double>(2 * L * (nmax + 1), 2 * L * (nmax + 1));
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			for (int j = 0; j < L; j++) {
				for (int m = 0; m <= nmax; m++) {
					if (i == j && n != m) {
						ddH[in(i, n, j, m)] = 0;
						ddnorm0[in(i, n, j, m)][0] = 0;
					} else {
						f = f0;
						fc = fc0;
						tabulate(f.begin() + i * (nmax + 1),
							f.begin() + (i + 1) * (nmax + 1), diff<double>(n));
						tabulate(fc.begin() + j * (nmax + 1),
							fc.begin() + (j + 1) * (nmax + 1), diff<double>(m));
						transform(fc.begin(), fc.end(), f.begin(),
							ddnorms[in(i, n, j, m)].begin(),
							multiplies<complex<double>>());
						reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(),
							ddnorms[in(i, n, j, m)].begin(), okeys.begin(),
							ddnormi[in(i, n, j, m)].begin());
						reduce_by_key(Lkeys.begin(), Lkeys.end(),
							ddnormi[in(i, n, j, m)].begin(), okeys.begin(),
							ddnorm0[in(i, n, j, m)].begin(), equal_to<int>(),
							multiplies<complex<double>>());
						for (int k = 0; k < L; k++) {
							ddnorm1[in(i, n, j, m)][k] =
								ddnorm0[in(i, n, j, m)][0]
									/ ddnormi[in(i, n, j, m)][k];
							ddnorm2[in(i, n, j, m)][k] =
								ddnorm1[in(i, n, j, m)][k]
									/ ddnormi[in(i, n, j, m)][mod(k + 1)];
							ddnorm3[in(i, n, j, m)][k] =
								ddnorm2[in(i, n, j, m)][k]
									/ ddnormi[in(i, n, j, m)][mod(k + 2)];
						}
						hamiltonian(fc, f, U0, dU, J, mu,
							ddnorm1[in(i, n, j, m)], ddnorm2[in(i, n, j, m)],
							ddnorm3[in(i, n, j, m)], U0p, Jp, H);
						ddH[in(i, n, j, m)] = H[0];
					}
					if (i == j) {
						ddcH[in(i, n, j, m)] = 0;
						ddcnorm0[in(i, n, j, m)][0] = 0;
					} else {
						f = f0;
						fc = fc0;
						tabulate(fc.begin() + i * (nmax + 1),
							fc.begin() + (i + 1) * (nmax + 1), diff<double>(n));
						tabulate(fc.begin() + j * (nmax + 1),
							fc.begin() + (j + 1) * (nmax + 1), diff<double>(m));
						transform(fc.begin(), fc.end(), f.begin(),
							ddcnorms[in(i, n, j, m)].begin(),
							multiplies<complex<double>>());
						reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(),
							ddcnorms[in(i, n, j, m)].begin(), okeys.begin(),
							ddcnormi[in(i, n, j, m)].begin());
						reduce_by_key(Lkeys.begin(), Lkeys.end(),
							ddcnormi[in(i, n, j, m)].begin(), okeys.begin(),
							ddcnorm0[in(i, n, j, m)].begin(), equal_to<int>(),
							multiplies<complex<double>>());
						for (int k = 0; k < L; k++) {
							ddcnorm1[in(i, n, j, m)][k] = ddcnorm0[in(i, n, j,
								m)][0] / ddcnormi[in(i, n, j, m)][k];
							ddcnorm2[in(i, n, j, m)][k] = ddcnorm1[in(i, n, j,
								m)][k] / ddcnormi[in(i, n, j, m)][mod(k + 1)];
							ddcnorm3[in(i, n, j, m)][k] = ddcnorm2[in(i, n, j,
								m)][k] / ddcnormi[in(i, n, j, m)][mod(k + 2)];
						}
						hamiltonian(fc, f, U0, dU, J, mu,
							ddcnorm1[in(i, n, j, m)], ddcnorm2[in(i, n, j, m)],
							ddcnorm3[in(i, n, j, m)], U0p, Jp, H);
						ddcH[in(i, n, j, m)] = H[0];
					}
					complex<double> ddHN = -(dH[in(i, n)] * dcnorm0[in(j, m)][0]
						+ dcH[in(j, m)] * dnorm0[in(i, n)][0])
						/ (norm00[0] * norm00[0])
						+ (2.0 * H0[0] * dnorm0[in(i, n)][0]
							* dcnorm0[in(j, m)][0])
							/ (norm00[0] * norm00[0] * norm00[0])
						+ ddH[in(i, n, j, m)] / norm00[0]
						- H0[0] * ddnorm0[in(i, n, j, m)][0]
							/ (norm00[0] * norm00[0]);
					complex<double> ddcHN = -(dcH[in(i, n)]
						* dcnorm0[in(j, m)][0]
						+ dcH[in(j, m)] * dcnorm0[in(i, n)][0])
						/ (norm00[0] * norm00[0])
						+ (2.0 * H0[0] * dcnorm0[in(i, n)][0]
							* dcnorm0[in(j, m)][0])
							/ (norm00[0] * norm00[0] * norm00[0])
						+ ddcH[in(i, n, j, m)] / norm00[0]
						- H0[0] * ddcnorm0[in(i, n, j, m)][0]
							/ (norm00[0] * norm00[0]);
					hess(2 * in(j, m), 2 * in(i, n)) = 2
						* (ddHN + ddcHN).real();
					hess(2 * in(j, m), 2 * in(i, n) + 1) = 2
						* (ddHN + ddcHN).imag();
					hess(2 * in(j, m) + 1, 2 * in(i, n)) = 2
						* (ddHN + ddcHN).imag();
					hess(2 * in(j, m) + 1, 2 * in(i, n) + 1) = 2
						* (ddHN - ddcHN).real();
				}
			}
		}
	}
}

double energy::value(const vector<double>& x) {
	host_vector<complex<double>> fh(L * (nmax + 1)), fch(L * (nmax + 1));
	for (int i = 0; i < L * (nmax + 1); i++) {
		fh[i] = complex<double>(x[2 * i], x[2 * i + 1]);
		fch[i] = complex<double>(x[2 * i], -x[2 * i + 1]);
	}

	state_type f = fh, fc = fch;

	int_vector keys(L * (nmax + 1)), okeys(L);

	auto nmaxrep = make_repeat_iterator(counting_iterator<int>(0), nmax + 1);
	copy(nmaxrep, nmaxrep + L * (nmax + 1), keys.begin());

	complex_vector norms(L * (nmax + 1));
	transform(fc.begin(), fc.end(), f.begin(), norms.begin(),
		multiplies<complex<double>>());

	complex_vector normi(L);
	reduce_by_key(keys.begin(), keys.end(), norms.begin(), okeys.begin(),
		normi.begin());

	auto Lrep = make_repeat_iterator(counting_iterator<int>(0), L);
	copy(Lrep, Lrep + L, keys.begin());

	complex_vector norm0(1);
	reduce_by_key(keys.begin(), keys.begin() + L, normi.begin(), okeys.begin(),
		norm0.begin(), equal_to<int>(), multiplies<complex<double>>());

	complex_vector norm1(L), norm2(L), norm3(L);
	for (int i = 0; i < L; i++) {
		norm1[i] = norm0[0] / normi[i];
		norm2[i] = norm1[i] / normi[mod(i + 1)];
		norm3[i] = norm2[i] / normi[mod(i + 2)];
	}

	state_type H(1);
	double_vector U0p(1, 0), Jp(L, 0);
	hamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);

	return H[0].real() / norm0[0].real();
}

void energy::gradient(const vector<double> &x, vector<double> &grad) {
	host_vector<complex<double>> fh(L * (nmax + 1)), fch(L * (nmax + 1));
	for (int i = 0; i < L * (nmax + 1); i++) {
		fh[i] = complex<double>(x[2 * i], x[2 * i + 1]);
		fch[i] = complex<double>(x[2 * i], -x[2 * i + 1]);
	}

	state_type f = fh, fc0 = fch;

	int_vector keys(L * (nmax + 1)), okeys(L);
	int_vector nmaxkeys(L * (nmax + 1));
	int_vector Lkeys(L);

	auto nmaxrep = make_repeat_iterator(counting_iterator<int>(0), nmax + 1);
	copy(nmaxrep, nmaxrep + L * (nmax + 1), nmaxkeys.begin());

	complex_vector norms(L * (nmax + 1));
	transform(fc0.begin(), fc0.end(), f.begin(), norms.begin(),
		multiplies<complex<double>>());

	complex_vector normi(L);
	reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), norms.begin(),
		okeys.begin(), normi.begin());

	auto Lrep = make_repeat_iterator(counting_iterator<int>(0), L);
	copy(Lrep, Lrep + L, Lkeys.begin());

	complex_vector norm0(1);
	reduce_by_key(Lkeys.begin(), Lkeys.end(), normi.begin(), okeys.begin(),
		norm0.begin(), equal_to<int>(), multiplies<complex<double>>());
	complex_vector norm00 = norm0;

	complex_vector norm1(L), norm2(L), norm3(L);
	for (int i = 0; i < L; i++) {
		norm1[i] = norm0[0] / normi[i];
		norm2[i] = norm1[i] / normi[mod(i + 1)];
		norm3[i] = norm2[i] / normi[mod(i + 2)];
	}

	state_type fc(L * (nmax + 1));
	fc = fc0;

	state_type H(1);
	double_vector U0p(1, 0), Jp(L, 0);
	hamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);
	state_type H0 = H;

	vector<double> dH(2 * L * (nmax + 1));
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			fc = fc0;
			tabulate(fc.begin() + i * (nmax + 1),
				fc.begin() + (i + 1) * (nmax + 1), diff<double>(n));
			transform(fc.begin(), fc.end(), f.begin(), norms.begin(),
				multiplies<complex<double>>());
			reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), norms.begin(),
				okeys.begin(), normi.begin());
			reduce_by_key(Lkeys.begin(), Lkeys.end(), normi.begin(),
				okeys.begin(), norm0.begin(), equal_to<int>(),
				multiplies<complex<double>>());
			for (int i = 0; i < L; i++) {
				norm1[i] = norm0[0] / normi[i];
				norm2[i] = norm1[i] / normi[mod(i + 1)];
				norm3[i] = norm2[i] / normi[mod(i + 2)];
			}
			hamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);
			complex<double> dH = H[0] / norm00[0]
				- H0[0] * norm0[0] / (norm00[0] * norm00[0]);
			grad[2 * in(i, n)] = 2 * dH.real();
			grad[2 * in(i, n) + 1] = 2 * dH.imag();
		}
	}
}
