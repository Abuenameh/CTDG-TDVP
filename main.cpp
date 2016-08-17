/*
 * main.cu
 *
 *  Created on: Aug 5, 2016
 *      Author: Abuenameh
 */

#include <iostream>
#include <array>
#include <vector>
#include <limits>
#include <random>
#include <functional>
#include <iterator>
#include <algorithm>
#include <ctime>

using std::function;
using std::array;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::ostream_iterator;
using std::vector;
using std::numeric_limits;
using std::mt19937;
using std::uniform_real_distribution;
using std::bind;
using std::generate;
using std::time;
using std::ref;
using std::exception;

#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

using thrust::device_vector;
using thrust::host_vector;
using thrust::complex;
using thrust::copy_n;
using thrust::tabulate;
using thrust::transform;
using thrust::reduce_by_key;
using thrust::equal_to;
using thrust::multiplies;
using thrust::conj;
using thrust::counting_iterator;
using thrust::divides;
using thrust::make_transform_iterator;
using thrust::minus;
using thrust::raw_pointer_cast;
using thrust::max_element;

#include <boost/lexical_cast.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>
#include <boost/numeric/odeint.hpp>

using boost::lexical_cast;
using boost::posix_time::ptime;
using boost::posix_time::time_period;
using boost::posix_time::microsec_clock;
using boost::filesystem::path;

//using namespace boost::posix_time;
//using namespace boost::filesystem;
using namespace boost::numeric::odeint;

#define EIGEN_DONT_VECTORIZE

//#include <cppoptlib/solver/lbfgssolver.h>
//
//using cppoptlib::LbfgsSolver;
//using cppoptlib::Criteria;

//#include <nlopt.hpp>
//
//using nlopt::opt;
//using nlopt::algorithm;

#include <dlib/optimization.h>

using dlib::matrix;
using dlib::find_min;
using dlib::lbfgs_search_strategy;
using dlib::objective_delta_stop_strategy;
using dlib::find_min_using_approximate_derivatives;
using dlib::derivative;
using dlib::find_min_trust_region;
using dlib::gradient_norm_stop_strategy;

#include "gutzwiller.hpp"
#include "dynamics.hpp"
#include "mathematica.hpp"
#include "groundstate.hpp"
#include "orderparameter.hpp"
#include "progress_bar.hpp"

typedef matrix<double, 0, 1> column_vector;

typedef function<array<double, L>(int, double)> SiteFunction;
typedef function<double(int, double)> SystemFunction;

const double M = 1000;
const double g13 = 2.5e9;
const double g24 = 2.5e9;
//const double delta = 1.0e12;
const double Delta = -2.0e10;
const double alpha = 1.1e7;

const double Ng = sqrt(M) * g13;

struct system_param {
	system_param(double x) :
		x(x) {
	}

	double operator()(int i, double t) {
		return x;
	}

	double x;
};

struct site_param {
	site_param(double y) {
		x.fill(y);
	}

	array<double, L> operator()(int i, double t) {
		return x;
	}

	array<double, L> x;
};

struct dU_param {
	array<double, L> operator()(int i, double t) {
		array<double, L> ret;
		ret.fill(0);
//		for (int i = 0; i < L; i++) {
//			ret[i] = 0.1 + 0.02 * i;
//		}
		return ret;
	}
};

double UW(double W) {
	return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W)
		/ ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

complex<double> dot(vector<complex<double>>&v, vector<complex<double>>&w) {
	complex<double> res = 0;
	for (int i = 0; i < v.size(); i++) {
		res += conj(v[i]) * w[i];
	}
	return res;
}

struct W0_param {
	W0_param(double W) :
		W(W) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> ret;
		ret.fill(W);
		return ret;
	}

	double W;
};

struct W_param {
	W_param(double Wi, double Wf, double tau) :
		Wi(Wi), Wf(Wf), tau(tau) {
	}

	array<double, L> operator()(int i, double t) {
		double Wx =
			(t < tau) ?
				Wi + (Wf - Wi) * t / tau : Wf + (Wi - Wf) * (t - tau) / tau;
		array<double, L> ret;
		ret.fill(Wx);
		return ret;
	}

	double Wi, Wf, tau;
};

struct Wp_param {
	Wp_param(double Wi, double Wf, double tau) :
		Wi(Wi), Wf(Wf), tau(tau) {
	}

	array<double, L> operator()(int i, double t) {
		double Wpx = (t < tau) ? (Wf - Wi) / tau : (Wi - Wf) / tau;
		array<double, L> ret;
		ret.fill(Wpx);
		return ret;
	}

	double Wi, Wf, tau;
};

template<typename T>
struct U0_param {

	U0_param(T& W) :
		W(W) {
	}

	double operator()(int i, double t) {
		double Wi = W(i, t)[0];
		return -2 * (g24 * g24) / Delta * (Ng * Ng * Wi * Wi)
			/ ((Ng * Ng + Wi * Wi) * (Ng * Ng + Wi * Wi));
	}

	T& W;
};

template<typename T, typename U>
struct U0p_param {

	U0p_param(T& W, U& Wp) :
		W(W), Wp(Wp) {
	}

	double operator()(int i, double t) {
		double Wi = W(i, t)[0];
		double Wip = Wp(i, t)[0];
		return 8 * pow(g24, 2) * pow(Ng, 2) * pow(Wi, 3) * Wip
			/ (Delta * pow(pow(Ng, 2) + pow(Wi, 2), 3))
			- 4 * pow(g24, 2) * pow(Ng, 2) * Wi * Wip
				/ (Delta * pow(pow(Ng, 2) + pow(Wi, 2), 2));
	}

	T& W;
	U& Wp;
};

template<typename T>
struct J_param {

	J_param(T& W) :
		W(W) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> Wi = W(i, t);
		array<double, L> ret;
		for (int i = 0; i < L; i++) {
			ret[i] = alpha * (Wi[i] * Wi[i]) / (Ng * Ng + Wi[i] * Wi[i]);
		}
		return ret;
	}

	T& W;
};

template<typename T, typename U>
struct Jp_param {

	Jp_param(T& W, U& Wp) :
		W(W), Wp(Wp) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> Wi = W(i, t);
		array<double, L> Wip = Wp(i, t);
		array<double, L> ret;
		for (int i = 0; i < L; i++) {
			int j = mod(i + 1);
			ret[i] = -alpha * pow(Wi[i], 2) * Wi[j] * Wip[i]
				/ (pow(pow(Ng, 2) + pow(Wi[i], 2), 1.5)
					* sqrt(pow(Ng, 2) + pow(Wi[j], 2)))
				+ alpha * Wi[j] * Wip[i]
					/ (sqrt(pow(Ng, 2) + pow(Wi[i], 2))
						* sqrt(pow(Ng, 2) + pow(Wi[j], 2)))
				- alpha * pow(Wi[j], 2) * Wi[i] * Wip[j]
					/ (pow(pow(Ng, 2) + pow(Wi[j], 2), 1.5)
						* sqrt(pow(Ng, 2) + pow(Wi[i], 2)))
				+ alpha * Wi[i] * Wip[j]
					/ (sqrt(pow(Ng, 2) + pow(Wi[i], 2))
						* sqrt(pow(Ng, 2) + pow(Wi[j], 2)));
		}
		return ret;
	}

	T& W;
	U& Wp;
};

struct push_back_state_and_time {
	vector<ode_state_type>& m_states;
	vector<double>& m_times;
	vector<double>& Es;
	ProgressBar& prog;
	double tf;
	W_param& W;
	double mu;

	push_back_state_and_time(vector<ode_state_type> &states,
		vector<double> &times, vector<double> &Es, ProgressBar& prog, double tf,
		W_param& W, double mu) :
		m_states(states), m_times(times), prog(prog), tf(tf), W(W), mu(mu), Es(
			Es) {
	}

	void operator()(const ode_state_type &x, double t) {
		U0_param<W_param> U0p(W);
		J_param<W_param> J0p(W);
		array<double, L> Ja = J0p(0, t);
		double U0t = U0p(0, t);
		host_vector<double> dUt(L, 0), Jt(L);
		for (int i = 0; i < L; i++) {
			Jt[i] = Ja[i];
		}

		energy en(U0t, dUt, Jt, mu);
		vector<double> xri(2 * L * (nmax + 1));
		for (int i = 0; i < L * (nmax + 1); i++) {
			xri[2 * i] = x[i].real();
			xri[2 * i + 1] = x[i].imag();
		}
		Es.push_back(en.value(xri));

		m_states.push_back(x);
		m_times.push_back(t);
		prog.Progressed(int(100 * (t / tf)));
	}
};

struct progress_observer {
	ProgressBar& prog;
	double tf;

	progress_observer(ProgressBar& prog, double tf) :
		prog(prog), tf(tf) {
	}

	void operator()(const ode_state_type &x, double t) {
		prog.Progressed(int(100 * (t / tf)));
	}
};

double objective(const vector<double>& x, vector<double>& grad, void* data) {
	energy* en = static_cast<energy*>(data);
	if (!grad.empty()) {
		en->gradient(x, grad);
	}
	return en->value(x);
}

class energy_model {
public:
	energy_model(energy& en) :
		en(en) {
	}

	double operator()(const column_vector& x) const {
		const vector<double> xv(x.begin(), x.end());
		return en.value(xv);
	}

	double objective(const column_vector& x) const {
		const vector<double> xv(x.begin(), x.end());
//		copy(x.begin(), x.end(), ostream_iterator<double>(cout, ","));
//		cout << endl;
//		cout << en.value(xv) << endl;
		return en.value(xv);
	}

	const column_vector gradient(const column_vector& x) const {
		const vector<double> xv(x.begin(), x.end());
		column_vector res(2 * L * (nmax + 1));
		vector<double> grad(2 * L * (nmax + 1));
		en.gradient(xv, grad);
		copy(grad.begin(), grad.end(), res.begin());
//		copy(res.begin(), res.end(), ostream_iterator<double>(cout,","));
//		cout << endl;
//				copy(grad.begin(), grad.end(), ostream_iterator<double>(cout,","));
//				cout << endl;
//				copy(res.begin(), res.end(), ostream_iterator<double>(cout,","));
//				cout << endl;
		return res;
	}

	const matrix<double> hessian(const column_vector& x) const {
		const vector<double> xv(x.begin(), x.end());
		matrix<double> hess(2 * L * (nmax + 1), 2 * L * (nmax + 1));
		column_vector grad(2 * L * (nmax + 1));
		en.get_derivative_and_hessian(x, grad, hess);
		return hess;
	}

private:
	energy& en;
};

double energy_obj(energy& en, column_vector& x) {
	const vector<double> xv(x.begin(), x.end());
	return en.value(xv);
}

//double energy_obj(energy& en) {
////	const vector<double> xv(x.begin(), x.end());
////	return en.value(xv);
//	return 0;
//}

//const column_vector energy_grad(energy& en, const column_vector& x) {
//	const vector<double> xv(x.begin(), x.end());
//	column_vector res(2*L*(nmax+1));
//	vector<double> grad(2*L*(nmax+1));
//	en.gradient(xv, grad);
//	copy(grad.begin(), grad.end(), res.begin());
//	return res;
//}

int main(int argc, char** argv) {

	mt19937 rng;
//	uniform_real_distribution<> uni(-pow(1./(nmax+1),1./(2*L)), pow(1./(nmax+1),1./(2*L)));
//	uniform_real_distribution<> uni(-sqrt(1./(nmax+1)), sqrt(1./(nmax+1)));
	uniform_real_distribution<> uni(-1, 1);
//	cout << sqrt(1./(nmax+1)) << endl;

	ptime begin = microsec_clock::local_time();

	int seed0 = lexical_cast<int>(argv[1]);
	int nseed = lexical_cast<int>(argv[2]);

	double Wi = lexical_cast<double>(argv[3]);
	double Wf = lexical_cast<double>(argv[4]);

	double mu = lexical_cast<double>(argv[5]);

	double D = lexical_cast<double>(argv[6]);

	double taui = lexical_cast<double>(argv[7]);
	double tauf = lexical_cast<double>(argv[8]);
	int ntaus = lexical_cast<int>(argv[9]);

	int resi = lexical_cast<int>(argv[10]);

//	double dt = lexical_cast<double>(argv[11]);

#ifdef AMAZON
	path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/CTDG TDVP");
#else
	path resdir("/Users/Abuenameh/Documents/Simulation Results/CTDG TDVP");
#endif
	if (!exists(resdir)) {
		cerr << "Results directory " << resdir << " does not exist!" << endl;
		exit(1);
	}
	ostringstream oss;
	oss << "res." << resi << ".txt";
	path resfile = resdir / oss.str();
	while (exists(resfile)) {
		resi++;
		oss.str("");
		oss << "res." << resi << ".txt";
		resfile = resdir / oss.str();
	}

//	vector<vector<double>> xis;
//	for (int i = 0; i < nseed; i++) {
//		rng.seed(seed + i);
//		vector<double> xi(L);
//		for (int j = 0; j < L; j++) {
//			xi[j] = (1 + D * uni(rng));
//		}
//		xis.push_back(xi);
//	}

	boost::filesystem::ofstream os(resfile);

	printMath(os, "seed0res", resi, seed0);
	printMath(os, "nseedres", resi, nseed);
	printMath(os, "Lres", resi, L);
	printMath(os, "nmaxres", resi, nmax);
	printMath(os, "Nres", resi, N);

	printMath(os, "Wires", resi, Wi);
	printMath(os, "Wfres", resi, Wf);
	printMath(os, "mures", resi, mu);
	os << flush;

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	vector<energy> en;

	vector<vector<double>> f0(N);
	for (int j = 0; j < N; j++) {

		int seed = seed0 + j;

		W0_param W0(Wi);
		U0_param<W0_param> U0p(W0);
		J_param<W0_param> J0p(W0);
		array<double, L> Ja = J0p(seed, 0);
		double U00 = U0p(seed, 0);
		host_vector<double> dU0(L, 0), J0(L);
		double mu0 = 0.5 * U00;
		for (int i = 0; i < L; i++) {
			J0[i] = Ja[i];
		}

//		en[j] = energy(U00, dU0, J0, mu0);
		en.push_back(energy(U00, dU0, J0, mu0));

		vector<double> f0j(2 * L * (nmax + 1));
		rng.seed(time(0));
		function<double()> rnd = bind(uni, rng);
		generate(f0j.begin(), f0j.end(), rnd);

		column_vector f0jd(2 * L * (nmax + 1));
		copy(f0j.begin(), f0j.end(), f0jd.begin());

		energy_model mod(en[j]);
		auto objbind = bind(&energy_model::objective, &mod,
			std::placeholders::_1);
		auto gradbind = bind(&energy_model::gradient, &mod,
			std::placeholders::_1);
		auto hessbind = bind(&energy_model::hessian, &mod,
			std::placeholders::_1);
		find_min(lbfgs_search_strategy(10),
			objective_delta_stop_strategy(1e-12), objbind, gradbind, f0jd,
			-1e12); //numeric_limits<double>::infinity());

//		find_min(newton_search_strategy(hessbind),
//			objective_delta_stop_strategy(1e-7), objbind, gradbind, f0jd,
//			-1e12);
//		find_min_trust_region(gradient_norm_stop_strategy(1e-10), en[j], f0jd,
//			10);

		copy(f0jd.begin(), f0jd.end(), f0j.begin());

//		opt lopt(algorithm::LD_LBFGS, 2 * L * (nmax + 1));
//		lopt.set_min_objective(objective, &en[j]);
//		lopt.set_lower_bounds(-1);
//		lopt.set_upper_bounds(1);
//		lopt.set_ftol_rel(1e-16);
//		lopt.set_ftol_abs(1e-16);
//
//		double E0j = 0;
//		try {
//			lopt.optimize(f0j, E0j);
//		} catch (std::exception& e) {
//			cerr << e.what() << endl;
//			exit(1);
//		}

//		double norm00 = 1;
		array<double, L> norm0i;
		for (int i = 0; i < L; i++) {
			norm0i[i] = 0;
			for (int n = 0; n <= nmax; n++) {
				norm0i[i] += pow(f0j[2 * (i * (nmax + 1) + n)], 2)
					+ pow(f0j[2 * (i * (nmax + 1) + n) + 1], 2);
			}
//		norm *= normi[i];
//			norm00 *= norm0i[i];
		}

		for (int i = 0; i < L; i++) {
			for (int n = 0; n <= nmax; n++) {
				f0j[2 * (i * (nmax + 1) + n)] /= sqrt(norm0i[i]);
				f0j[2 * (i * (nmax + 1) + n) + 1] /= sqrt(norm0i[i]);
			}
		}
		f0[j] = f0j;
	}

	ode_state_type f0c(Ndim);

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < L * (nmax + 1); i++) {
			f0c[j * L * (nmax + 1) + i] = std::complex<double>(f0[j][2 * i],
				f0[j][2 * i + 1]);
		}
	}

	vector<ode_state_type> fs;
	vector<double> ts;
	vector<double> Es;
	vector<ode_state_type> dfs;

	vector<double> taus, steps;
	vector<vector<double>> Eis(N), Efs(N), Qs(N), ps(N);
	vector<vector<vector<complex<double>>> > b0s(N), bfs(N);
	vector<vector<vector<vector<complex<double>>> >> f0s(N), ffs(N);
	vector<bool> failed;

	for (int itau = 0; itau < ntaus; itau++) {

		ProgressBar prog(100, std::to_string(itau).c_str());

		double tau =
			(ntaus == 1) ? taui : taui + itau * (tauf - taui) / (ntaus - 1);
		taus.push_back(tau);

		double tf = 2 * tau;

		W_param Wt(Wi, Wf, tau);
		U0_param<W_param> U0t(Wt);
		J_param<W_param> Jt(Wt);
		Wp_param Wpt(Wi, Wf, tau);
		U0p_param<W_param, Wp_param> U0pt(Wt, Wpt);
		Jp_param<W_param, Wp_param> Jpt(Wt, Wpt);

		ode_state_type f = f0c;
//		for (int i = 0; i < L; i++) {
//			for (int n = 0; n <= nmax; n++) {
//				f[in(i,n)] = complex<double>(0.5+0.01*(i+1)+0.002*n, 0.6+0.02*(i+1)+0.001*n);
//			}
//		}

		dynamics dyn(cublas_handle, solver_handle, U0t, site_param(0), Jt,
			system_param(0.5 * UW(Wi)), U0pt, Jpt);

		bool fail = false;
		size_t stepsi = 0;
		try {
			stepsi = integrate_adaptive(bulirsch_stoer<ode_state_type>(), dyn, f,
				0., tf, tf / 10, progress_observer(prog, tf));
		} catch (exception& e) {
			fail = true;
		}
		failed.push_back(fail);

		steps.push_back(stepsi);

//		array<double, L> normi;
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < L; i++) {
				double normi = 0;
//				normi[j * L + i] = 0;
//				normi[i] = 0;
				for (int n = 0; n <= nmax; n++) {
//					normi[j * L + i] += norm(
//						f[j * L * (nmax + 1) + i * (nmax + 1) + n]);
					normi += norm(f[j * L * (nmax + 1) + i * (nmax + 1) + n]);
				}
				for (int n = 0; n <= nmax; n++) {
					f[j * L * (nmax + 1) + i * (nmax + 1) + n] /= sqrt(normi);
				}
			}
		}
//		for (int j = 0; j < N; j++) {
//			for (int i = 0; i < L; i++) {
//				for (int n = 0; n <= nmax; n++) {
//					f[j * L * (nmax + 1) + i * (nmax + 1) + n] /= sqrt(
//						normi[j * L + i]);
//				}
//			}
//		}

		vector<double> ff(2 * Ndim);
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < L; i++) {
				for (int n = 0; n <= nmax; n++) {
					ff[2 * (j * L * (nmax + 1) + i * (nmax + 1) + n)] = f[j * L
						* (nmax + 1) + i * (nmax + 1) + n].real();
					ff[2 * (j * L * (nmax + 1) + i * (nmax + 1) + n) + 1] = f[j
						* L * (nmax + 1) + i * (nmax + 1) + n].imag();
				}
			}
		}

		vector<double> Ei(N), Ef(N), Q(N);
		for (int j = 0; j < N; j++) {
			Ei[j] = en[j].value(f0[j]);
			Eis[j].push_back(Ei[j]);
			vector<double> ffj(2 * L * (nmax + 1));
			for (int i = 0; i < 2 * L * (nmax + 1); i++) {
				ffj[i] = ff[j * 2 * L * (nmax + 1) + i];
			}
//			printMath(os, "ff", resi, ffj);
			Ef[j] = en[j].value(ffj);
			Efs[j].push_back(Ef[j]);
			Q[j] = (Ef[j] - Ei[j]) / U0t(0, 0);
			Qs[j].push_back(Q[j]);
		}
//		cout << "Ediff% = " << ::math((Ei[0]-Ei[1])/Ei[0]) << endl;
//		cout << ::math(Q) << endl;

		vector<double> p(N);
		vector<vector<complex<double>>> b0(N), bf(N);
		for (int j = 0; j < N; j++) {
			double U00 = U0t(j, 0);
			double U0f = U0t(j, tf);
			array<double, L> J0a = Jt(j, 0);
			vector<double> J0(J0a.begin(), J0a.end());
			array<double, L> Jfa = Jt(j, tf);
			vector<double> Jf(Jfa.begin(), Jfa.end());

			vector<vector<complex<double>>> f0c(L), ffc(L);
			vector<double> pi(L);
			p[j] = 0;
			for (int i = 0; i < L; i++) {
				vector<complex<double>> f0i(nmax + 1);
				for (int n = 0; n <= nmax; n++) {
					f0i[n] = complex<double>(f0[j][2 * (i * (nmax + 1) + n)],
						f0[j][2 * (i * (nmax + 1) + n) + 1]);
				}
				f0c[i] = f0i;
				vector<complex<double>> ffi(nmax + 1);
				for (int n = 0; n <= nmax; n++) {
					ffi[n] = complex<double>(
						ff[2 * (j * L * (nmax + 1) + i * (nmax + 1) + n)],
						ff[2 * (j * L * (nmax + 1) + i * (nmax + 1) + n) + 1]);
				}
				ffc[i] = ffi;
				pi[i] = 1 - norm(dot(ffi, f0i));
				p[j] += pi[i];
			}
			p[j] /= L;
			ps[j].push_back(p[j]);
			for (int i = 0; i < L; i++) {
				b0[j].push_back(b(f0c, i, J0, U00));
				bf[j].push_back(b(ffc, i, Jf, U0f));
			}
			b0s[j].push_back(b0[j]);
			bfs[j].push_back(bf[j]);
			f0s[j].push_back(f0c);
			ffs[j].push_back(ffc);
		}

	}

	cusolverDnDestroy(solver_handle);
	cublasDestroy(cublas_handle);

	printMath(os, "taures", resi, taus);
	printMath(os, "Eires", resi, Eis);
	printMath(os, "Efres", resi, Efs);
	printMath(os, "Qres", resi, Qs);
	printMath(os, "pres", resi, ps);
	printMath(os, "b0res", resi, b0s);
	printMath(os, "bfres", resi, bfs);
	printMath(os, "f0res", resi, f0s);
	printMath(os, "ffres", resi, ffs);
	printMath(os, "failed", resi, failed);

//	printMath(os, "tsres", resi, ts);
//	printMath(os, "fsres", resi, fs);
//	printMath(os, "dfsres", resi, dfs);
//	printMath(os, "Esres", resi, Es);
	printMath(os, "stepsres", resi, steps);

	ptime end = microsec_clock::local_time();
	time_period period(begin, end);
	cout << endl << period.length() << endl << endl;

	os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;

	return 0;
}

