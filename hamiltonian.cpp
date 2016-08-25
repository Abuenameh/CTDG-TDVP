/*
 * hamiltonian.cu
 *
 *  Created on: Aug 5, 2016
 *      Author: Abuenameh
 */

#include <thrust/tuple.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/gather.h>

using thrust::get;
using thrust::tuple;
using thrust::complex;
using thrust::device_vector;
using thrust::host_vector;
using thrust::counting_iterator;
using thrust::make_counting_iterator;
using thrust::make_tuple;
using thrust::make_zip_iterator;
//using thrust::iterator_adaptor;
//using thrust::use_default;
using thrust::constant_iterator;
using thrust::multiplies;
using thrust::plus;
using thrust::placeholders::_1;
using thrust::fill;
using thrust::for_each_n;
using thrust::copy_n;
using thrust::iterator_adaptor;
using thrust::use_default;
using thrust::iterator_core_access;
using thrust::sequence;
using thrust::tabulate;
using thrust::gather;

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

inline int mod(int i, int L) {
	return (i + L) % L;
}

__device__ __host__ inline double g(int n, int m) {
	return sqrt(1.0 * (n + 1) * m);
}

struct hamiltonian_functor {

	template<class T>
	__device__ __host__
	void operator()(T x) {

#include "vars.incl"

		complex<double> H = 0;

#include "energy.incl"

//		printf("Hi = (%f)+I*(%f)\n", H.real(), H.imag());
		get<9>(x) += H;
	}

};

struct canonical_functor {

	template<class T>
	__device__ __host__
	void operator()(T x) {

#include "vars.incl"

		complex<double> iSp = 0;

#include "canonical.incl"

		get<9>(x) += iSp;
	}

};

void hamiltonian(state_type& fc, state_type& f, const double_vector& U0,
	const double_vector& dU, const double_vector& J, const double_vector& mu,
	complex_vector& norm1, complex_vector& norm2, complex_vector& norm3, const double_vector U0p, const double_vector& Jp,
	state_type& H) {

	int N = H.size();
	int L = dU.size() / N;
	int nmax = (fc.size() / N / L) - 1;

	fill(H.begin(), H.end(), 0);

//	for (int i = 0; i < L; i++) {
//		int_vector perm(N * L), fperm(N * L * (nmax + 1));
//		for (int j = 0; j < L; j++) {
//			for (int k = 0; k < N; k++) {
//				perm[k * L + j] = (j + i) % L + k * L;
//			}
//			for (int n = 0; n <= nmax; n++) {
//				for (int k = 0; k < N; k++) {
//					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
//						* ((j + i) % L) + n + k * L * (nmax + 1);
//				}
//			}
//		}
//#include "zip.incl"
//		for_each_n(zip, N, hamiltonian_functor());
//	}

	state_type Hi(N*L);
	fill(Hi.begin(), Hi.end(), 0);
	int_vector Hkeys(N * L);

	state_type fci(N*L*(nmax+1)*L), fi(N*L*(nmax+1)*L);
	double_vector U0i(N*L), dUi(N*L*L), Ji(N*L*L), mui(N*L), U0pi(N*L), Jpi(N*L*L);
	complex_vector norm1i(N*L*L), norm2i(N*L*L), norm3i(N*L*L);
	for (int i = 0; i < L; i++) {
		int_vector perm(N * L), fperm(N * L * (nmax + 1));
		for (int j = 0; j < L; j++) {
			for (int k = 0; k < N; k++) {
				perm[k * L + j] = (j + i) % L + k * L;
			}
			for (int n = 0; n <= nmax; n++) {
				for (int k = 0; k < N; k++) {
					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
						* ((j + i) % L) + n + k * L * (nmax + 1);
				}
			}
		}
		copy_n(make_permutation_iterator(fc.begin(), fperm.begin()), N*L*(nmax+1), fci.begin()+i*N*L*(nmax+1));
		copy_n(make_permutation_iterator(f.begin(), fperm.begin()), N*L*(nmax+1), fi.begin()+i*N*L*(nmax+1));
		copy_n(U0.begin(), N, U0i.begin()+i*N);
		copy_n(make_permutation_iterator(dU.begin(), perm.begin()), N*L, dUi.begin()+i*N*L);
		copy_n(make_permutation_iterator(J.begin(), perm.begin()), N*L, Ji.begin()+i*N*L);
		copy_n(mu.begin(), N, mui.begin()+i*N);
		copy_n(U0p.begin(), N, U0pi.begin()+i*N);
		copy_n(make_permutation_iterator(Jp.begin(), perm.begin()), N*L, Jpi.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm1.begin(), perm.begin()), N*L, norm1i.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm2.begin(), perm.begin()), N*L, norm2i.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm3.begin(), perm.begin()), N*L, norm3i.begin()+i*N*L);
		copy_n(make_counting_iterator(0), N, Hkeys.begin()+i*N);
	}

#include "zip.incl"
	for_each_n(zip, N*L, hamiltonian_functor());

	int_vector okeys(N*L);
	reduce_by_key(Hkeys.begin(), Hkeys.end(), Hi.begin(), okeys.begin(), H.begin());

//	for (int k = 0; k < N; k++) {
//		for (int i = 0; i < L; i++) {
//			int_vector perm(N * L), fperm(N * L * (nmax + 1));
//			for (int j = 0; j < L; j++) {
////				perm[k * L + j] = (j + i) % L + k * L;
//				perm[k * L + j] = mod(j + i, L) + k * L;
//				for (int n = 0; n <= nmax; n++) {
////					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
////						* ((j + i) % L) + n + k * L * (nmax + 1);
//					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
//						* (mod(j + i, L)) + n + k * L * (nmax + 1);
//				}
//			}
//#include "zip.incl"
//			for_each_n(zip, N, hamiltonian_functor());
//		}
//	}
}

void dynamicshamiltonian(state_type& fc, state_type& f, const double_vector& U0,
	const double_vector& dU, const double_vector& J, const double_vector& mu,
	complex_vector& norm1, complex_vector& norm2, complex_vector& norm3, const double_vector U0p, const double_vector& Jp,
	state_type& H) {

	int N = H.size();
	int L = dU.size() / N;
	int nmax = (fc.size() / N / L) - 1;

	state_type iSp(N);

	fill(H.begin(), H.end(), 0);
	fill(iSp.begin(), iSp.end(), 0);
/*
	for (int i = 0; i < L; i++) {
		int_vector perm(N * L), fperm(N * L * (nmax + 1));
		for (int j = 0; j < L; j++) {
			for (int k = 0; k < N; k++) {
				perm[k * L + j] = (j + i) % L + k * L;
			}
			for (int n = 0; n <= nmax; n++) {
				for (int k = 0; k < N; k++) {
					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
						* ((j + i) % L) + n + k * L * (nmax + 1);
				}
			}
		}
#include "zip.incl"
		for_each_n(zip, N, hamiltonian_functor());
		for_each_n(zip, N, canonical_functor());
	}
	constant_iterator<complex<double>> I(complex<double>(0,1));
	state_type mSp(N);
	transform(iSp.begin(), iSp.end(), I, mSp.begin(), multiplies<complex<double>>());
	transform(H.begin(), H.end(), mSp.begin(), H.begin(), plus<complex<double>>());
*/

	state_type Hi(N*L);
	int_vector Lmap(N * L);

	state_type fci(N*L*(nmax+1)*L), fi(N*L*(nmax+1)*L);
	double_vector U0i(N*L), dUi(N*L*L), Ji(N*L*L), mui(N*L), U0pi(N*L), Jpi(N*L*L);
	complex_vector norm1i(N*L*L), norm2i(N*L*L), norm3i(N*L*L);
	for (int i = 0; i < L; i++) {
		int_vector perm(N * L), fperm(N * L * (nmax + 1));
		for (int j = 0; j < L; j++) {
			for (int k = 0; k < N; k++) {
				perm[k * L + j] = (j + i) % L + k * L;
			}
			for (int n = 0; n <= nmax; n++) {
				for (int k = 0; k < N; k++) {
					fperm[k * L * (nmax + 1) + j * (nmax + 1) + n] = (nmax + 1)
						* ((j + i) % L) + n + k * L * (nmax + 1);
				}
			}
		}
		copy_n(make_permutation_iterator(fc.begin(), fperm.begin()), N*L*(nmax+1), fci.begin()+i*N*L*(nmax+1));
		copy_n(make_permutation_iterator(f.begin(), fperm.begin()), N*L*(nmax+1), fi.begin()+i*N*L*(nmax+1));
		copy_n(U0.begin(), N, U0i.begin()+i*N);
		copy_n(make_permutation_iterator(dU.begin(), perm.begin()), N*L, dUi.begin()+i*N*L);
		copy_n(make_permutation_iterator(J.begin(), perm.begin()), N*L, Ji.begin()+i*N*L);
		copy_n(mu.begin(), N, mui.begin()+i*N);
		copy_n(U0p.begin(), N, U0pi.begin()+i*N);
		copy_n(make_permutation_iterator(Jp.begin(), perm.begin()), N*L, Jpi.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm1.begin(), perm.begin()), N*L, norm1i.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm2.begin(), perm.begin()), N*L, norm2i.begin()+i*N*L);
		copy_n(make_permutation_iterator(norm3.begin(), perm.begin()), N*L, norm3i.begin()+i*N*L);
//		sequence(Lmap.begin()+i*N, Lmap.begin()+(i+1)*N, i, L);
//		copy_n(make_counting_iterator(0), N, Hkeys.begin()+i*N);
	}
	for (int i = 0; i < N; i++) {
		sequence(Lmap.begin()+i*L, Lmap.begin()+(i+1)*L, i, N);
	}

//	printf("U0 = ");
//	for (int i = 0; i < U0.size(); i++) {
//		printf("%f,", U0[i]);
//	}
//	printf("\n");
//	printf("U0i = ");
//	for(int i = 0; i < N*L; i++) {
//		printf("%f,", U0i[i]);
//	}
//	printf("\n");
//
//	printf("Lmap = ");
//	for (int i = 0; i < N*L; i++) {
//	printf("%d,", Lmap[i]);
//	}
//	printf("\n");

	int_vector Hkeys(N*L);
	tabulate(Hkeys.begin(), Hkeys.end(), _1/L);
//	printf("Hkeys = ");
//	for (int i = 0; i < N*L; i++) {
//	printf("%d,", Hkeys[i]);
//	}
//	printf("\n");

#include "zip.incl"
	fill(Hi.begin(), Hi.end(), 0);
	for_each_n(zip, N*L, hamiltonian_functor());
	state_type Ei(N*L);
	gather(Lmap.begin(), Lmap.end(), Hi.begin(), Ei.begin());
//	copy(Hi.begin(), Hi.end(), Ei.begin());
//	printf("Ei = ");
//	for (int i = 0; i < Ei.size(); i++) {
//	printf("(%f)+I*(%f),", Ei[i].real(), Ei[i].imag());
//	}
//	printf("\n");
//
//	exit(0);
//	printf("H = ");
//	for (int i = 0; i < H.size(); i++) {
//	printf("(%f)+I*(%f),", H[i].real(), H[i].imag());
//	}
//	printf("\n");

	fill(Hi.begin(), Hi.end(), 0);
	for_each_n(zip, N*L, canonical_functor());
	state_type iSpi(N*L);
	gather(Lmap.begin(), Lmap.end(), Hi.begin(), iSpi.begin());
//	copy(Hi.begin(), Hi.end(), iSpi.begin());
//	printf("iSpi = ");
//	for (int i = 0; i < N*L; i++) {
//	printf("(%f)+I*(%f),", iSpi[i].real(), iSpi[i].imag());
//	}
//	printf("\n");
//
//	exit(0);

	int_vector okeys(N*L);
//	printf("Ei.size() = %lu\n", Ei.size());
//	for(int i = 0; i < H.size(); i++) {
//		H[i] = 0;
////		complex<double> Eii = 0;
//		for (int j = 0; j < L; j++) {
//			H[i] += Ei[j*N+i];
//		}
////		H[i] = Ei[i];
//	}
////	printf("H1 = ");
//	for (int i = 0; i < H.size(); i++) {
////	printf("(%f)+I*(%f),", H[i].real(), H[i].imag());
//	}
//	double qwe = H[0].real();
//	double asd = H[0].imag();
//	printf("%f,%f\n",qwe,asd);
//	printf("%f,%f\n",H[0].real(),H[0].imag());
//	printf("%f,%f,%f,%f\n",H[0].real(),H[0].imag(),H[1].real(),H[1].imag());
//	printf("%f,%f,%f,%f,%f,%f\n",H[0].real(),H[0].imag(),H[1].real(),H[1].imag(),H[2].real(),H[2].imag());
//	printf("\n");
//	fill(H.begin(), H.end(), 0);
//	int_vector okeys2(2*N*L);
//	fill(okeys2.begin(), okeys2.end(), 100);
//	state_type H2(N*L);
//	fill(H2.begin(), H2.end(), 42.);

	reduce_by_key(Hkeys.begin(), Hkeys.end(), Ei.begin(), okeys.begin(), H.begin());

//	printf("H = ");
//	for (int i = 0; i < H.size(); i++) {
//	printf("(%f)+I*(%f),", H[i].real(), H[i].imag());
//	}
//	printf("\n");
//	printf("okeys = ");
//	for (int i = 0; i < okeys2.size(); i++) {
//	printf("%d,", okeys2[i]);
//	}
//	printf("\n");
	reduce_by_key(Hkeys.begin(), Hkeys.end(), iSpi.begin(), okeys.begin(), iSp.begin());

	constant_iterator<complex<double>> I(complex<double>(0,1));
	state_type mSp(N);
	transform(iSp.begin(), iSp.end(), I, mSp.begin(), multiplies<complex<double>>());
	transform(H.begin(), H.end(), mSp.begin(), H.begin(), plus<complex<double>>());
}
