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
#include <thread>
#include <queue>
#include <string>

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
using std::queue;
using std::mutex;
using std::lock_guard;
using std::string;
using std::thread;

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

using namespace boost::numeric::odeint;

#define EIGEN_DONT_VECTORIZE

#define MAKESTRING2(a) #a
#define MAKESTRING(a) MAKESTRING2(a)

#define eigen_assert(x) \
	  if (!(x)) { throw (std::runtime_error(MAKESTRING(x))); }

#include <Eigen/Core>

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

template<typename T>
class locking_queue {
public:
	void push(T const & value) {
		lock_guard<mutex> lock(mut);
		queue.push(value);
	}

	bool pop(T & value) {
		lock_guard<mutex> lock(mut);
		if (queue.empty()) {
			return false;
		} else {
			value = queue.front();
			queue.pop();
			return true;
		}
	}

private:
	queue<T> queue;
	mutex mut;
};

struct W0_param {

	W0_param() {
	}

	W0_param(double W) :
		W(W) {
		array<double, L> xi;
		xi.fill(1);
		xis.push_back(xi);
	}

	W0_param(double W, vector<array<double, L>> xis) :
		W(W), xis(xis) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			ret[j] = W * xis[i][j];
		}
		return ret;
	}

private:
	vector<array<double, L>> xis;
	double W;
};

struct W_param {

	W_param() {
	}

	W_param(double Wi, double Wf, double tau) :
		Wi(Wi), Wf(Wf), tau(tau) {
		array<double, L> xi;
		xi.fill(1);
		xis.push_back(xi);
	}

	W_param(double Wi, double Wf, double tau, vector<array<double, L>> xis) :
		Wi(Wi), Wf(Wf), tau(tau), xis(xis) {
	}

	array<double, L> operator()(int i, double t) {
		double Wx =
			(t < tau) ?
				Wi + (Wf - Wi) * t / tau : Wf + (Wi - Wf) * (t - tau) / tau;
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			ret[j] = Wx * xis[i][j];
		}
		return ret;
	}

private:
	vector<array<double, L>> xis;
	double Wi, Wf, tau;
};

struct Wp_param {

	Wp_param() {
	}

	Wp_param(double Wi, double Wf, double tau) :
		Wi(Wi), Wf(Wf), tau(tau) {
		array<double, L> xi;
		xi.fill(1);
		xis.push_back(xi);
	}

	Wp_param(double Wi, double Wf, double tau, vector<array<double, L>> xis) :
		Wi(Wi), Wf(Wf), tau(tau), xis(xis) {
	}

	array<double, L> operator()(int i, double t) {
		double Wpx = (t < tau) ? (Wf - Wi) / tau : (Wi - Wf) / tau;
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			ret[j] = Wpx * xis[i][j];
		}
		return ret;
	}

private:
	vector<array<double, L>> xis;
	double Wi, Wf, tau;
};

template<typename T>
struct U0_param {

	U0_param() {
	}

	U0_param(T W) :
		W(W) {
	}

	double operator()(int i, double t) {
		double Wi = W(i, t)[0];
		return -2 * (g24 * g24) / Delta * (Ng * Ng * Wi * Wi)
			/ ((Ng * Ng + Wi * Wi) * (Ng * Ng + Wi * Wi));
	}

private:
	T W;
};

template<typename T, typename U>
struct U0p_param {

	U0p_param() {
	}

	U0p_param(T W, U Wp) :
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

private:
	T W;
	U Wp;
};

template<typename T, typename U>
struct dU_param {

	dU_param() {
	}

	dU_param(T W0, U W) :
		W0(W0), W(W) {
	}

	array<double, L> operator()(int i, double t) {
		double W0i = W0(0, t)[0];
		double U0 = UW(W0i);
		array<double, L> Wi = W(i, t);
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			ret[j] = -2 * (g24 * g24) / Delta * (Ng * Ng * Wi[j] * Wi[j])
				/ ((Ng * Ng + Wi[j] * Wi[j]) * (Ng * Ng + Wi[j] * Wi[j])) - U0;
		}
		return ret;
	}

private:
	T W0;
	U W;
};

template<typename T>
struct J_param {

	J_param() {
	}

	J_param(T W) :
		W(W) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> Wi = W(i, t);
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			int k = mod(j + 1);
			ret[j] = alpha * (Wi[j] * Wi[k]) / (Ng * Ng + Wi[j] * Wi[k]);
		}
		return ret;
	}

private:
	T W;
};

template<typename T, typename U>
struct Jp_param {

	Jp_param() {
	}

	Jp_param(T W, U Wp) :
		W(W), Wp(Wp) {
	}

	array<double, L> operator()(int i, double t) {
		array<double, L> Wi = W(i, t);
		array<double, L> Wip = Wp(i, t);
		array<double, L> ret;
		for (int j = 0; j < L; j++) {
			int k = mod(j + 1);
			ret[i] = -alpha * pow(Wi[j], 2) * Wi[k] * Wip[j]
				/ (pow(pow(Ng, 2) + pow(Wi[j], 2), 1.5)
					* sqrt(pow(Ng, 2) + pow(Wi[k], 2)))
				+ alpha * Wi[k] * Wip[j]
					/ (sqrt(pow(Ng, 2) + pow(Wi[j], 2))
						* sqrt(pow(Ng, 2) + pow(Wi[k], 2)))
				- alpha * pow(Wi[k], 2) * Wi[j] * Wip[k]
					/ (pow(pow(Ng, 2) + pow(Wi[k], 2), 1.5)
						* sqrt(pow(Ng, 2) + pow(Wi[j], 2)))
				+ alpha * Wi[j] * Wip[k]
					/ (sqrt(pow(Ng, 2) + pow(Wi[j], 2))
						* sqrt(pow(Ng, 2) + pow(Wi[k], 2)));
		}
		return ret;
	}

private:
	T W;
	U Wp;
};

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
		return en.value(xv);
	}

	const column_vector gradient(const column_vector& x) const {
		const vector<double> xv(x.begin(), x.end());
		column_vector res(2 * L * (nmax + 1));
		vector<double> grad(2 * L * (nmax + 1));
		en.gradient(xv, grad);
		copy(grad.begin(), grad.end(), res.begin());
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

template<class T, class U>
struct taupoint {

	taupoint() {
	}

	taupoint(int itau, double tau, T& dUW, T& W, U& Wp, double mu,
		ode_state_type f0) :
		itau(itau), tau(tau), U0(dUW), dU(dUW, W), J(W), U0p(W, Wp), Jp(W, Wp), mu(
			mu), f0(f0) {

	}

	int itau;
	double tau;
	U0_param<T> U0;
	dU_param<T, T> dU;
	J_param<T> J;
	U0p_param<T, U> U0p;
	Jp_param<T, U> Jp;
	double mu;
	ode_state_type f0;
};

struct tauresults {

	tauresults(vector<vector<double>>& Eis, vector<vector<double>>& Efs,
		vector<vector<double>>& Qs, vector<vector<double>>& ps,
		vector<vector<vector<complex<double>>> >& b0s, vector<vector<vector<complex<double>>> >& bfs, vector<vector<vector<vector<complex<double>>> >>& f0s, vector<vector<vector<vector<complex<double>>> >>& ffs, vector<bool>& failed, vector<string>& errors, vector<int>& steps) : Eis(Eis), Efs(Efs), Qs(Qs), ps(ps), b0s(b0s), bfs(bfs), f0s(f0s), ffs(ffs), failed(failed), errors(errors), steps(steps) {}

	vector<vector<double>>& Eis;
	vector<vector<double>>& Efs;
	vector<vector<double>>& Qs;
	vector<vector<double>>& ps;
	vector<vector<vector<complex<double>>> >& b0s;
	vector<vector<vector<complex<double>>> >& bfs;
	vector<vector<vector<vector<complex<double>>> >>& f0s;
	vector<vector<vector<vector<complex<double>>> >>& ffs;
	vector<bool>& failed;
	vector<string>& errors;
	vector<int>& steps;
};

mutex resultsmutex;
mutex progmutex;

void threadfunc(vector<energy>& en, vector<vector<double>>& f0,
	locking_queue<taupoint<W_param, Wp_param>>& points, tauresults& results,
	ProgressBar& prog) {
	taupoint<W_param, Wp_param> point;
	while (points.pop(point)) {
		int itau = point.itau;
		double tau = point.tau;

		ode_state_type f = point.f0;

		dynamics dyn(point.U0, point.dU, point.J, system_param(point.mu),
			point.U0p, point.Jp);

		double tf = 2 * tau;

		bool fail = false;
		size_t steps = 0;
		string error = "";
		try {
			steps = integrate_adaptive(bulirsch_stoer<ode_state_type>(), dyn, f,
				0., tf, tf / 10);
		} catch (exception& e) {
			error = e.what();
			fail = true;
		}

		for (int j = 0; j < N; j++) {
			for (int i = 0; i < L; i++) {
				double normi = 0;
				for (int n = 0; n <= nmax; n++) {
					normi += norm(f[j * L * (nmax + 1) + i * (nmax + 1) + n]);
				}
				for (int n = 0; n <= nmax; n++) {
					f[j * L * (nmax + 1) + i * (nmax + 1) + n] /= sqrt(normi);
				}
			}
		}

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
			vector<double> ffj(2 * L * (nmax + 1));
			for (int i = 0; i < 2 * L * (nmax + 1); i++) {
				ffj[i] = ff[j * 2 * L * (nmax + 1) + i];
			}
			Ef[j] = en[j].value(ffj);
			Q[j] = (Ef[j] - Ei[j]) / point.U0(0, 0);
		}

		vector<double> p(N);
		vector<vector<complex<double>>> b0(N), bf(N);
		vector<vector<vector<complex<double>>> > f0s(N), ffs(N);
		for (int j = 0; j < N; j++) {
			double U00 = point.U0(j, 0);
			double U0f = point.U0(j, tf);
			array<double, L> J0a = point.J(j, 0);
			vector<double> J0(J0a.begin(), J0a.end());
			array<double, L> Jfa = point.J(j, tf);
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
			for (int i = 0; i < L; i++) {
				b0[j].push_back(b(f0c, i, J0, U00));
				bf[j].push_back(b(ffc, i, Jf, U0f));
			}
			f0s[j] = f0c;
			ffs[j] = ffc;
		}
		{
			lock_guard<mutex> resultslock(resultsmutex);
			for (int j = 0; j < N; j++) {
				results.Eis[j][itau] = Ei[j];
				results.Efs[j][itau] = Ef[j];
				results.Qs[j][itau] = Q[j];
				results.ps[j][itau] = p[j];
				results.b0s[j][itau] = b0[j];
				results.bfs[j][itau] = bf[j];
				results.f0s[j][itau] = f0s[j];
				results.ffs[j][itau] = ffs[j];
			}
			results.failed[itau] = fail;
			results.errors[itau] = error;
			results.steps[itau] = steps;
		}

		{
			lock_guard<mutex> proglock(progmutex);
			++prog;
		}
	}
}

int main(int argc, char** argv) {

	Eigen::initParallel();

	mt19937 rng;
	uniform_real_distribution<> uni(-1, 1);

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

	int nthreads = lexical_cast<int>(argv[10]);

	int resi = lexical_cast<int>(argv[11]);

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

	vector<array<double, L>> xis;
	for (int i = 0; i < nseed; i++) {
		rng.seed(seed0 + i);
		array<double, L> xi;
		for (int j = 0; j < L; j++) {
			xi[j] = (1 + D * uni(rng));
		}
		xis.push_back(xi);
	}

	vector<array<double, L>> xi0(1);
	xi0[0].fill(1);

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

	vector<energy> en;

	vector<vector<double>> f0(N);
	for (int j = 0; j < N; j++) {

		int seed = seed0 + j;

		W0_param W0(Wi, xis);
		W0_param dUW0(Wi, xi0);
		U0_param<W0_param> U0p(dUW0);
		dU_param<W0_param, W0_param> dU0p(dUW0, W0);
		J_param<W0_param> J0p(W0);
		array<double, L> Ja = J0p(j, 0);
		array<double, L> dUa = dU0p(j, 0);
		double U00 = U0p(j, 0);
		host_vector<double> dU0(L), J0(L);
		double mu0 = mu * U00;
		for (int i = 0; i < L; i++) {
			dU0[i] = dUa[i];
			J0[i] = Ja[i];
		}

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
			-1e12);

		copy(f0jd.begin(), f0jd.end(), f0j.begin());

		array<double, L> norm0i;
		for (int i = 0; i < L; i++) {
			norm0i[i] = 0;
			for (int n = 0; n <= nmax; n++) {
				norm0i[i] += pow(f0j[2 * (i * (nmax + 1) + n)], 2)
					+ pow(f0j[2 * (i * (nmax + 1) + n) + 1], 2);
			}
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

	vector<double> taus;
	vector<vector<double>> Eis(N, vector<double>(ntaus)), Efs(N,
		vector<double>(ntaus)), Qs(N, vector<double>(ntaus)), ps(N,
		vector<double>(ntaus));
	vector<vector<vector<complex<double>>> > b0s(N, vector<vector<complex<double>>>(ntaus)), bfs(N, vector<vector<complex<double>>>(ntaus));
	vector<vector<vector<vector<complex<double>>> >> f0s(N, vector<vector<vector<complex<double>>>>(ntaus)), ffs(N, vector<vector<vector<complex<double>>>>(ntaus));
	vector<bool> failed(ntaus);
	vector<string> errors(ntaus);
	vector<int> steps(ntaus);

	locking_queue<taupoint<W_param, Wp_param>> points;

	ProgressBar prog(ntaus);
	prog.Progressed(0);

	for (int itau = 0; itau < ntaus; itau++) {

		double tau =
			(ntaus == 1) ? taui : taui + itau * (tauf - taui) / (ntaus - 1);
		taus.push_back(tau);

		double tf = 2 * tau;

		W_param Wt(Wi, Wf, tau, xis);
		W_param dUWt(Wi, Wf, tau, xi0);
		U0_param<W_param> U0t(dUWt);
		dU_param<W_param, W_param> dUt(dUWt, Wt);
		J_param<W_param> Jt(Wt);
		Wp_param Wpt(Wi, Wf, tau, xis);
		U0p_param<W_param, Wp_param> U0pt(Wt, Wpt);
		Jp_param<W_param, Wp_param> Jpt(Wt, Wpt);

		taupoint<W_param, Wp_param> point(itau, tau, dUWt, Wt, Wpt, mu, f0c);
		points.push(point);

	}

	tauresults results(Eis, Efs, Qs, ps, b0s, bfs, f0s, ffs, failed, errors,
		steps);

	vector<thread> threads;
	for (int i = 0; i < nthreads; i++) {
		threads.emplace_back(threadfunc, ref(en), ref(f0), ref(points),
			ref(results), ref(prog));
	}
	for (int i = 0; i < nthreads; i++) {
		threads[i].join();
	}

	printMath(os, "taures", resi, taus);
	printMath(os, "Eires", resi, Eis);
	printMath(os, "Efres", resi, Efs);
	printMath(os, "Qres", resi, Qs);
	printMath(os, "pres", resi, ps);
	printMath(os, "b0res", resi, b0s);
	printMath(os, "bfres", resi, bfs);
	printMath(os, "f0res", resi, f0s);
	printMath(os, "ffres", resi, ffs);
	printMath(os, "failed", resi, results.failed);
	printMath(os, "errors", resi, results.errors);

	printMath(os, "stepsres", resi, results.steps);

	ptime end = microsec_clock::local_time();
	time_period period(begin, end);
	cout << endl << period.length() << endl << endl;

	os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;

	return 0;
}

