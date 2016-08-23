/*
 * dynamics.cu
 *
 *  Created on: Aug 5, 2016
 *      Author: Abuenameh
 */

#include <iostream>
#include <limits>
#include <iterator>
#include <iomanip>
#include <fstream>

using std::cout;
using std::endl;
using std::ostream_iterator;
using std::numeric_limits;
using std::ostream;
using std::ostringstream;
using std::setprecision;
using std::ofstream;

#include <boost/algorithm/string.hpp>

using boost::algorithm::replace_all_copy;

#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/count.h>

using thrust::counting_iterator;
using thrust::iterator_adaptor;
using thrust::use_default;
using thrust::iterator_core_access;
using thrust::equal_to;
using thrust::multiplies;
using thrust::divides;
using thrust::minus;
using thrust::tabulate;
using thrust::max_element;
using thrust::plus;

#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include <Eigen/SPQRSupport>
//#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>

using Eigen::MatrixXcd;
using Eigen::VectorXcd;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::CompleteOrthogonalDecomposition;
using Eigen::BDCSVD;
using Eigen::SparseMatrix;
// using Eigen::SPQR;
using Eigen::VectorXi;
// using Eigen::SparseQR;
using Eigen::COLAMDOrdering;
using Eigen::AMDOrdering;
using Eigen::NaturalOrdering;
using Eigen::Lower;
using Eigen::Matrix;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

#include "gutzwiller.hpp"
#include "dynamics.hpp"

typedef Matrix<std::complex<double>, nmax + 1, nmax + 1> GramMatrix;
typedef Matrix<std::complex<double>, nmax + 1, 1> SiteVector;

extern void hamiltonian(state_type& fc, state_type& f, const double_vector& U0,
	const double_vector& dU, const double_vector& J, const double_vector& mu,
	complex_vector& norm1, complex_vector& norm2, complex_vector& norm3,
	state_type& H);

extern void dynamicshamiltonian(state_type& fc, state_type& f,
	const double_vector& U0, const double_vector& dU, const double_vector& J,
	const double_vector& mu, complex_vector& norm1, complex_vector& norm2,
	complex_vector& norm3, const double_vector U0p, const double_vector& Jp,
	state_type& H);

template<typename Iterator>
class strided_range {
public:

	typedef typename thrust::iterator_difference<Iterator>::type difference_type;

	struct stride_functor: public thrust::unary_function<difference_type,
		difference_type> {
		difference_type stride;

		stride_functor(difference_type stride) :
			stride(stride) {
		}

		__host__ __device__
		difference_type operator()(const difference_type& i) const {
			return stride * i;
		}
	};

	typedef typename thrust::counting_iterator<difference_type> CountingIterator;
	typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
	typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

	// type of the strided_range iterator
	typedef PermutationIterator iterator;

	// construct strided_range for the range [first,last)
	strided_range(Iterator first, Iterator last, difference_type stride) :
		first(first), last(last), stride(stride) {
	}

	iterator begin(void) const {
		return PermutationIterator(first,
			TransformIterator(CountingIterator(0), stride_functor(stride)));
	}

	iterator end(void) const {
		return begin() + ((last - first) + (stride - 1)) / stride;
	}

protected:
	Iterator first;
	Iterator last;
	difference_type stride;
};

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

template<typename T>
class mathematic {
public:

	mathematic(T& v_) :
		v(v_) {
	}

	T& v;
};

template<>
class mathematic<double> {
public:

	mathematic(double d_) :
		d(d_) {
	}

	double d;
};

template<>
class mathematic<std::complex<double> > {
public:

	mathematic(std::complex<double> c_) :
		c(c_) {
	}

	std::complex<double> c;
};

mathematic<double> mathe(double d) {
	return mathematic<double>(d);
}

mathematic<std::complex<double> > mathe(std::complex<double> c) {
	return mathematic<std::complex<double> >(c);
}

ostream& operator<<(ostream& out, const mathematic<double> m) {
	double d = m.d;
	ostringstream oss;
	oss << setprecision(numeric_limits<double>::digits10) << d;
	out << replace_all_copy(oss.str(), "e", "*^");
	return out;
}

ostream& operator<<(ostream& out, const mathematic<std::complex<double> > m) {
	std::complex<double> c = m.c;
	out << "(" << mathematic<double>(c.real()) << ")+I("
		<< mathematic<double>(c.imag()) << ")";
	return out;
}

void dynamics::operator()(const ode_state_type& fcon, ode_state_type& dfdt,
	const double t) {

	state_type f0(Ndim);
	thrust::copy(fcon.begin(), fcon.end(), f0.begin());

	state_type fc0(Ndim);
	transform(f0.begin(), f0.end(), fc0.begin(), conjop<double>());

	host_vector<complex<double>> f = f0;

	int N = f.size() / L / (nmax + 1);

	host_vector<double> U0h(N), dU(N*L), J(N*L), mu(N), U0p(N), Jp(N*L);
	for (int i = 0; i < N; i++) {
		U0h[i] = U0f(i, t);
		copy_n(dUf(i, t).begin(), L, dUh.begin() + i * L);
		copy_n(Jf(i, t).begin(), L, Jh.begin() + i * L);
		muh[i] = muf(i, t);
		U0p[i] = U0pf(i, t);
		copy_n(Jpf(i, t).begin(), L, Jp.begin() + i * L);
	}
	double_vector U0 = U0h, dU = dUh, J = Jh, mu = muh, U0p = U0ph, Jp = Jph;

	int_vector okeys(N * L);
	int_vector nmaxkeys(N * L * (nmax + 1));
	int_vector Lkeys(N * L);

	auto nmaxrep = make_repeat_iterator(counting_iterator<int>(0), nmax + 1);
	copy(nmaxrep, nmaxrep + N * L * (nmax + 1), nmaxkeys.begin());

	complex_vector norms(Ndim);
	transform(fc0.begin(), fc0.end(), f.begin(), norms.begin(),
		multiplies<complex<double>>());

	complex_vector normi(N * L);
	reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), norms.begin(),
		okeys.begin(), normi.begin());

	auto Lrep = make_repeat_iterator(counting_iterator<int>(0), L);
	copy(Lrep, Lrep + N * L, Lkeys.begin());

	complex_vector norm0(N);
	reduce_by_key(Lkeys.begin(), Lkeys.begin() + N * L, normi.begin(),
		okeys.begin(), norm0.begin(), equal_to<int>(),
		multiplies<complex<double>>());

	host_vector<complex<double>> norm0h = norm0, normih = normi;
	host_vector<complex<double>> norm1h(N*L), norm2h(N*L), norm3h(N*L);
//	complex_vector norm1(N * L), norm2(N * L), norm3(N * L);
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < N; j++) {
			norm1h[j * L + i] = norm0h[j] / normih[j * L + i];
			norm2h[j * L + i] = norm1h[j * L + i] / normih[j * L + mod(i + 1)];
			norm3h[j * L + i] = norm2h[j * L + i] / normih[j * L + mod(i + 2)];
		}
	}
	complex_vector norm1 = norm1h, norm2 = norm2h, norm3 = norm3h;

	host_vector<complex<double>> fc = fc0;
/*
	state_type H(N);

	host_vector<double> U0h = U0;
	host_vector<double> dUh = dU;
	host_vector<double> Jh = J;
	host_vector<double> muh = mu;
	host_vector<complex<double>> norm1h2 = norm1;
	host_vector<complex<double>> norm2h2 = norm2;
	host_vector<complex<double>> norm3h2 = norm3;
	dynamicshamiltonian(fc, f, U0, dU, J, mu, norm1, norm2, norm3, U0p, Jp, H);
	state_type E = H;

	complex_vector dH(Ndim);
	complex_vector dnorms(Ndim);
	complex_vector dnormi(N * L);
	complex_vector dnorm0(N);
	complex_vector dnorm1(N * L), dnorm2(N * L), dnorm3(N * L);
	complex_vector covariant(Ndim);
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			f = f0;
			fc = fc0;
			for (int k = 0; k < N; k++) {
				tabulate(fc.begin() + k * L * (nmax + 1) + i * (nmax + 1),
					fc.begin() + k * L * (nmax + 1) + (i + 1) * (nmax + 1),
					diff<double>(n));
			}
			transform(fc.begin(), fc.end(), f.begin(), dnorms.begin(),
				multiplies<complex<double>>());
			reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), dnorms.begin(),
				okeys.begin(), dnormi.begin());
			reduce_by_key(Lkeys.begin(), Lkeys.end(), dnormi.begin(),
				okeys.begin(), dnorm0.begin(), equal_to<int>(),
				multiplies<complex<double>>());
			for (int k = 0; k < N; k++) {
				covariant[in(k, i, n)] = dnorm0[k];
				for (int j = 0; j < L; j++) {
					dnorm1[k * L + j] = dnorm0[k] / dnormi[k * L + j];
					dnorm2[k * L + j] = dnorm1[k * L + j]
						/ dnormi[k * L + mod(j + 1)];
					dnorm3[k * L + j] = dnorm2[k * L + j]
						/ dnormi[k * L + mod(j + 2)];
				}
			}
			dynamicshamiltonian(fc, f, U0, dU, J, mu, dnorm1, dnorm2, dnorm3,
				U0p, Jp, H);
			strided_range<state_type::iterator> stride(dH.begin() + in(i, n),
				dH.end(), L * (nmax + 1));
			copy(H.begin(), H.end(), stride.begin());
		}
	}

	auto norm1rep = make_repeat_iterator(norm1.begin(), nmax + 1);

	auto norm0rep = make_repeat_iterator(norm0.begin(), L * (nmax + 1));

	auto Erep = make_repeat_iterator(E.begin(), L * (nmax + 1));

	state_type Hi1(Ndim);
	transform(dH.begin(), dH.end(), norm0rep, Hi1.begin(),
		divides<complex<double>>());

	state_type Hi2(Ndim);
	transform(covariant.begin(), covariant.end(), Erep, Hi2.begin(),
		multiplies<complex<double>>());

	state_type Hi3(Ndim);
	complex_vector norm0sq(N * L * (nmax + 1));
	transform(norm0rep, norm0rep + N * L * (nmax + 1), norm0sq.begin(),
		square<complex<double>>());
	transform(Hi2.begin(), Hi2.end(), norm0sq.begin(), Hi3.begin(),
		divides<complex<double>>());

	state_type Hi(Ndim);
	transform(Hi1.begin(), Hi1.end(), Hi3.begin(), Hi.begin(),
		minus<complex<double>>());
	host_vector<complex<double>> Hih = Hi;

	host_vector<complex<double>> fh = f;
	host_vector<complex<double>> norm0h = norm0;
	host_vector<complex<double>> norm1h = norm1;
	host_vector<complex<double>> normih = normi;

	host_vector<complex<double>> covarianth = covariant;
	complex_vector ddnorms(Ndim);
	complex_vector ddnormi(N * L);
	complex_vector ddnorm0(N);
	MatrixXcd Gij = MatrixXcd::Zero(Ndim, Ndim);
//	SparseMatrix<std::complex<double>> Gij(Ndim, Ndim);
//	Gij.reserve(VectorXi::Constant(Ndim, nmax+1));
//	for (int k = 0; k < N; k++) {
	for (int i = 0; i < L; i++) {
		for (int n = 0; n <= nmax; n++) {
			for (int m = 0; m <= nmax; m++) {
				fc = fc0;
				for (int k = 0; k < N; k++) {
					tabulate(fc.begin() + k * L * (nmax + 1) + i * (nmax + 1),
						fc.begin() + k * L * (nmax + 1) + (i + 1) * (nmax + 1),
						diff<double>(n));
				}
				f = f0;
				for (int k = 0; k < N; k++) {
					tabulate(f.begin() + k * L * (nmax + 1) + i * (nmax + 1),
						f.begin() + k * L * (nmax + 1) + (i + 1) * (nmax + 1),
						diff<double>(m));
				}
				transform(fc.begin(), fc.end(), f.begin(), ddnorms.begin(),
					multiplies<complex<double>>());
				reduce_by_key(nmaxkeys.begin(), nmaxkeys.end(), ddnorms.begin(),
					okeys.begin(), ddnormi.begin());
				reduce_by_key(Lkeys.begin(), Lkeys.end(), ddnormi.begin(),
					okeys.begin(), ddnorm0.begin(), equal_to<int>(),
					multiplies<complex<double>>());
				for (int k = 0; k < N; k++) {
					Gij(in(k, i, n), in(k, i, m)) = std::complex<double>(
						ddnorm0[k] / norm0h[k]
							- covarianth[in(k, i, n)]
								* conj(covarianth[in(k, i, m)])
								/ (norm0h[k] * norm0h[k]));
//							Gij.insert(in(k, i, n), in(k, i, m)) = std::complex<double>(1, 0)*(std::complex<double>(ddnorm0[k]/norm0h[k] - covarianth[in(k, i, n)]
//							             									* conj(covarianth[in(k, i, m)])
//							             									/ (norm0h[k] * norm0h[k])));
				}
			}
		}
	}
//		Gij.makeCompressed();

	VectorXcd Hiv(Ndim);
	for (int i = 0; i < Ndim; i++) {
		Hiv[i] = Hih[i];
	}

	VectorXcd dfdtv = Gij.completeOrthogonalDecomposition().solve(Hiv);

	for (int i = 0; i < Ndim; i++) {
		dfdt[i] = -std::complex<double>(0, 1) * dfdtv[i];
	}*/
}
