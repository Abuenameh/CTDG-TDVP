/*
 * mathematica.hpp
 *
 *  Created on: Aug 8, 2016
 *      Author: Abuenameh
 */

#ifndef MATHEMATICA_HPP_
#define MATHEMATICA_HPP_

#include <iostream>
#include <iomanip>
#include <complex>
#include <string>
#include <limits>

using std::ostream;
using std::ostringstream;
using std::string;
using std::setprecision;
using std::numeric_limits;

#include <boost/algorithm/string.hpp>

using boost::algorithm::replace_all_copy;

#include <thrust/complex.h>

template<typename T>
class mathematica {
public:

    mathematica(T& v_) : v(v_) {
    }

    T& v;
};

template<>
class mathematica<int> {
public:

    mathematica(int i_) : i(i_) {
    }

    int i;
};

template<>
class mathematica<const int> {
public:

    mathematica(int i_) : i(i_) {
    }

    int i;
};

template<>
class mathematica<double> {
public:

    mathematica(double d_) : d(d_) {
    }

    double d;
};

template<>
class mathematica<bool> {
public:

    mathematica(bool b_) : b(b_) {
    }

    bool b;
};

template<>
class mathematica<std::complex<double> > {
public:

    mathematica(std::complex<double> c_) : c(c_) {
    }

    std::complex<double> c;
};

template<>
class mathematica<thrust::complex<double> > {
public:

    mathematica(thrust::complex<double> c_) : c(c_) {
    }

    thrust::complex<double> c;
};

template<typename T>
mathematica<T> math(T& t) {
    return mathematica<T > (t);
}

mathematica<double> math(double d) {
    return mathematica<double>(d);
}

mathematica<bool> math(bool b) {
    return mathematica<bool>(b);
}

mathematica<std::complex<double> > math(std::complex<double> c) {
    return mathematica<std::complex<double> >(c);
}

mathematica<thrust::complex<double> > math(thrust::complex<double> c) {
    return mathematica<thrust::complex<double> >(c);
}

ostream& operator<<(ostream& out, const mathematica<int> m) {
    out << m.i;
    return out;
}

ostream& operator<<(ostream& out, const mathematica<const int> m) {
    out << m.i;
    return out;
}

ostream& operator<<(ostream& out, const mathematica<double> m) {
    double d = m.d;
    ostringstream oss;
    oss << setprecision(numeric_limits<double>::digits10) << d;
    out << replace_all_copy(oss.str(), "e", "*^");
    return out;
}

ostream& operator<<(ostream& out, const mathematica<bool> m) {
    out << (m.b ? "True" : "False");
    return out;
}

ostream& operator<<(ostream& out, const mathematica<std::complex<double> > m) {
    std::complex<double> c = m.c;
    out << "(" << mathematica<double>(c.real()) << ")+I(" << mathematica<double>(c.imag()) << ")";
    return out;
}

ostream& operator<<(ostream& out, const mathematica<thrust::complex<double> > m) {
    std::complex<double> c = m.c;
    out << "(" << mathematica<double>(c.real()) << ")+I(" << mathematica<double>(c.imag()) << ")";
    return out;
}

ostream& operator<<(ostream& out, const mathematica<string> m) {
    out << "\"" << m.v << "\"";
    return out;
}

template<typename T>
ostream& operator<<(ostream& out, const mathematica<T>& m) {
	T t = m.v;
	auto begin = t.begin();
	auto end = t.end();
	out << "{";
	if (begin != end) {
		out << math(*begin);
		for (auto it = begin+1; it < end; it++) {
			out << "," << math(*it);
		}
	}
	out << "}";
	return out;
}

template<typename T> void printMath(ostream& out, string name, T& t) {
    out << name << "=" << ::math(t) << ";" << std::endl;
}

template<typename T> void printMath(ostream& out, string name, int i, T& t) {
    out << name << "[" << i << "]" << "=" << ::math(t) << ";" << std::endl;
}

template<typename T> void printMath(ostream& out, string name, int i, int j, T& t) {
    if (j == -1) {
    out << name << "[" << i << "]" << "=" << ::math(t) << ";" << std::endl;
    }
    else {
    out << name << "[" << i << "," << j << "]" << "=" << ::math(t) << ";" << std::endl;
    }
}



#endif /* MATHEMATICA_HPP_ */
