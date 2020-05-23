#include <random>
#include <omp.h>
#include <ctime>
#include <cassert>

// command to compile this:
// clang++ -shared -fPIC -o lib.so lib.cc -std=c++17 -O3 -ffast-math -lomp -march=native -Xpreprocessor -fopenmp -lpthread

#include "lib.h"

#include <iostream>
#define LOG(what) fprintf(stderr, "%s %s = %f\n", __FUNCTION__, #what, (double)(what));

// std::mt19937_64 engine(1235 + omp_get_thread_num());
std::ranlux48_base engine(1235 + omp_get_thread_num() + std::time(nullptr));
#pragma omp threadprivate(engine)

// constexpr int n_curves = 300;
// constexpr int n_steps = 10000;

real_t compute_fraction_below(
    const real_t avg_win,  // 1-based !!
    const real_t avg_loss, // 1-based !!
    const real_t minimum_wealth,        // fraction !!
    const real_t max_drawdown_risk [[maybe_unused]],     // fraction !!   // should be removed, used only in the compute_b_via_mc_plus_cpp
    const real_t predicted_probability, // fraction !!
    const real_t b,
    const int n_curves,
    const int n_steps
) {
	int num = 0;
	const real_t win_size = (b*avg_win + (1-b));
	const real_t loss_size = (b*avg_loss + (1-b));
	#pragma omp parallel for reduction(+:num) schedule(static)
	for (int curve_idx = 0; curve_idx < n_curves; ++curve_idx) {
		std::discrete_distribution<int> ddistr({predicted_probability, 1.-predicted_probability});
		real_t ath = 1.;
		real_t current = 1.;
		real_t threshold = current * minimum_wealth;
		for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
			int event = ddistr(engine);
			if (event == 0) {
				current *= win_size;
			} else { // event == 1
				current *= loss_size;
			}
			if (current > ath) {
				ath = current;
				threshold = current * minimum_wealth;
			}
			if (current <= threshold) {
				// #pragma omp critical
				num += 1;
				break;
			}
		}
	}
	return real_t(num) / real_t(n_curves);
}

extern "C" real_t compute_b_via_mc_plus_cpp(
    const real_t avg_win,  // 1-based !!
    const real_t avg_loss, // 1-based !!
    const real_t minimum_wealth,        // fraction !!
    const real_t max_drawdown_risk,     // fraction !!
    const real_t predicted_probability,       // fraction !!
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
) {
	// avg_win = 1 + avg_win;
	// avg_loss = 1 - std::abs(avg_loss);

	real_t a = 0.;
	real_t b = size_upper_bound;

	{
		// special case when the size_upper_bound is too tight
		real_t frac = compute_fraction_below(
			avg_win, avg_loss, minimum_wealth, max_drawdown_risk, predicted_probability, b, n_curves, n_steps
		);
		if (frac < max_drawdown_risk) {
			return b;
		}
	}

	// double n_sim = 0.;

	while (b - a > xtol) {
		const real_t x = (a + b) / 2.;

		// ++n_sim;
		const real_t frac = compute_fraction_below(
			avg_win, avg_loss, minimum_wealth, max_drawdown_risk, predicted_probability, x, n_curves, n_steps
		);
		const real_t diff = frac - max_drawdown_risk;
		const real_t abs_diff = std::abs(diff);
		if (abs_diff < xtol) {
			// LOG(n_sim);
			return x;
		} else if (diff > 0) {
			b = x;
		} else { // diff < 0
			a = x;
		}
	}
	// LOG(n_sim);
	real_t x = (a + b) / 2.;
	return x;
}



real_t compute_fraction_below_non_compounded(
    const real_t avg_win,  // 0-based !!
    const real_t avg_loss, // 0-based !!
    const real_t minimum_wealth,        // fraction !!
    const real_t max_drawdown_risk [[maybe_unused]],     // fraction !!   // should be removed, used only in the compute_b_via_mc_plus_cpp
    const real_t predicted_probability, // fraction !!
    const real_t b,
    const int n_curves,
    const int n_steps
) {
	int num = 0;
	constexpr double initial_wealth = 1.;
	const real_t win_size = initial_wealth * b*avg_win;
	const real_t loss_size = initial_wealth * b*avg_loss;
	#pragma omp parallel for reduction(+:num) schedule(static)
	for (int curve_idx = 0; curve_idx < n_curves; ++curve_idx) {
		std::discrete_distribution<int> ddistr({predicted_probability, 1.-predicted_probability});
		real_t ath = 1.;
		real_t current = 1.;
		real_t threshold = current * minimum_wealth;
		for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
			int event = ddistr(engine);
			if (event == 0) {
				current += win_size;
			} else { // event == 1
				current -= loss_size;
			}
			if (current > ath) {
				ath = current;
				threshold = current * minimum_wealth;
			}
			if (current <= threshold) {
				// #pragma omp critical
				num += 1;
				break;
			}
		}
	}
	return real_t(num) / real_t(n_curves);
}

extern "C" real_t compute_b_via_mc_plus_cpp_non_compounded(
    const real_t avg_win,  // 0-based !!
    const real_t avg_loss, // 0-based !!
    const real_t minimum_wealth,        // fraction !!
    const real_t max_drawdown_risk,     // fraction !!
    const real_t predicted_probability,       // fraction !!
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
) {
	assert(avg_loss >= 0);

	real_t a = 0.;
	real_t b = size_upper_bound;

	// {
	// 	// special case when the size_upper_bound is too tight
	// 	real_t frac = compute_fraction_below_non_compounded(
	// 		avg_win, avg_loss, minimum_wealth, max_drawdown_risk, predicted_probability, b, n_curves, n_steps
	// 	);
	// 	if (frac < max_drawdown_risk) {
	// 		return b;
	// 	}
	// }

	// double n_sim = 0.;

	while (b - a > xtol) {
		const real_t x = (a + b) / 2.;

		// ++n_sim;
		const real_t frac = compute_fraction_below_non_compounded(
			avg_win, avg_loss, minimum_wealth, max_drawdown_risk, predicted_probability, x, n_curves, n_steps
		);
		const real_t diff = frac - max_drawdown_risk;
		const real_t abs_diff = std::abs(diff);
		if (abs_diff < xtol) {
			// LOG(n_sim);
			return x;
		} else if (diff > 0) {
			b = x;
		} else { // diff < 0
			a = x;
		}
	}
	// LOG(n_sim);
	real_t x = (a + b) / 2.;
	return x;
}

extern "C" real_t compute_bet_old(
    const real_t avg_win,  /* 0-based */
    const real_t avg_loss, /* 0-based */
    const real_t minimum_wealth,
    const real_t max_drawdown_risk,
    bool compounded,
    const real_t predicted_probability,
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
) {
	const real_t avg_win_nonneg = std::abs(avg_win);
	const real_t avg_loss_nonneg = std::abs(avg_loss);
	assert(0 <= predicted_probability && predicted_probability <= 1);
	assert(n_curves > 0);
	assert(n_steps > 0);

	if (compounded == true) {
		return compute_b_via_mc_plus_cpp(
		    1+avg_win_nonneg,
		    1-avg_loss_nonneg,
		    minimum_wealth,
		    max_drawdown_risk,
		    predicted_probability,
		    size_upper_bound,
		    xtol,
		    n_curves,
		    n_steps
		);
	} else { /* compounded == false */
		return compute_b_via_mc_plus_cpp_non_compounded(
		    avg_win_nonneg,
		    avg_loss_nonneg,
		    minimum_wealth,
		    max_drawdown_risk,
		    predicted_probability,
		    size_upper_bound,
		    xtol,
		    n_curves,
		    n_steps
		);
	}
}


/**************************************************************************************************/


template<bool compounded>
real_t compute_fraction_below_machinery(
    const real_t avg_win,  /* this is 0-based */
    const real_t avg_loss,
    const real_t minimum_wealth,
    const real_t predicted_probability,
    const real_t b,
    const int n_curves,
    const int n_steps
) {
	int num = 0;
	const real_t win_size = b*avg_win + 1*compounded;
	const real_t loss_size = -b*avg_loss + 1*compounded;
	#pragma omp parallel for reduction(+:num) schedule(static)
	for (int curve_idx = 0; curve_idx < n_curves; ++curve_idx) {
		std::discrete_distribution<int> ddistr({predicted_probability, 1.-predicted_probability});
		real_t ath = 1.;
		real_t current = 1.;
		real_t threshold = current * minimum_wealth;
		for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
			int event = ddistr(engine);
			if constexpr (compounded == true) {
				if (event == 0) {
					current *= win_size;
				} else { /* event == 1 */
					current *= loss_size;
				}
			} else { /* compounded == false */
				if (event == 0) {
					current += win_size;
				} else { /* event == 1 */
					current += loss_size;
				}
			}
			if (current > ath) {
				ath = current;
				threshold = current * minimum_wealth;
			}
			if (current <= threshold) {
				num += 1;
				break;
			}
		}
	}
	return real_t(num) / real_t(n_curves);
}

template<bool compounded>
real_t compute_bet_machinery(
    const real_t avg_win,
    const real_t avg_loss,
    const real_t minimum_wealth,
    const real_t max_drawdown_risk,
    const real_t predicted_probability,
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
) {
	real_t a = 0.;
	real_t b = size_upper_bound;

	while (b - a > xtol) {
		const real_t x = (a + b) / 2.;
		const real_t frac = compute_fraction_below_machinery<compounded>(
			avg_win, avg_loss, minimum_wealth, predicted_probability, x, n_curves, n_steps
		);
		const real_t diff = frac - max_drawdown_risk;
		const real_t abs_diff = std::abs(diff);
		if (abs_diff < xtol) {
			return x;
		} else if (diff > 0) {
			b = x;
		} else { /* diff < 0 */
			a = x;
		}
	}
	return (a + b) / 2.;
}

extern "C" real_t compute_bet(
    const real_t avg_win,  /* 0-based */
    const real_t avg_loss, /* 0-based */
    const real_t minimum_wealth,
    const real_t max_drawdown_risk,
    const bool compounded,
    const real_t predicted_probability,
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
) {
	const real_t avg_win_nonneg = std::abs(avg_win);
	const real_t avg_loss_nonneg = std::abs(avg_loss);
	assert(0 <= predicted_probability && predicted_probability <= 1);
	assert(n_curves > 0);
	assert(n_steps > 0);

	if (compounded == true) {
		return compute_bet_machinery<true>(
		    avg_win_nonneg,
		    avg_loss_nonneg,
		    minimum_wealth,
		    max_drawdown_risk,
		    predicted_probability,
		    size_upper_bound,
		    xtol,
		    n_curves,
		    n_steps
		);
	} else { /* compounded == false */
		return compute_bet_machinery<false>(
		    avg_win_nonneg,
		    avg_loss_nonneg,
		    minimum_wealth,
		    max_drawdown_risk,
		    predicted_probability,
		    size_upper_bound,
		    xtol,
		    n_curves,
		    n_steps
		);
	}
}