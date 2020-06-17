#include <vector>
#include <array>
#include <map>
#include <thread>
#include <cmath>
#include <limits>
#include <algorithm>
#include <future>

#include <cstdio>

#ifndef NDEBUG
#define _printf(...) printf(__VA_ARGS__);
#else
#define _printf(...)
#endif

extern "C" void triple_barrier_baseline(
	double* close,
	const int n_bars,
	int* vertical_barrier,
	double* unit_width,
	double* pt_mul,
	double* sl_mul,
	bool* event,
	int* end
) {
	int n_events;
	for (int bar_idx = 0, event_idx = 0; bar_idx < n_bars; ++bar_idx) {
		if (event[bar_idx]) {
			end[event_idx] = std::min(n_bars-1, vertical_barrier[bar_idx]);
			++event_idx;
			++n_events;
		}
	}
	std::vector<std::multimap<const double, int>::iterator> it_below(n_events);
	std::vector<std::multimap<const double, int>::iterator> it_above(n_events);
	std::multimap<const double, int> closing_above;
	std::multimap<const double, int> closing_below;
	for (int bar_idx = 0, event_idx = 0; bar_idx < n_bars; ++bar_idx) {
		// _printf("bar #%bar_idx\n", bar_idx);
		// _printf("close: %f\n", close[bar_idx]);
		// _printf("closing_above ");
		// for (auto it = closing_above.begin(); it != closing_above.end(); ++it) {
		// 	_printf("(%f, %bar_idx)  ", it->first, it->second);
		// }
		// _printf("\nclosing_below ");
		// for (auto it = closing_below.begin(); it != closing_below.end(); ++it) {
		// 	_printf("(%f, %bar_idx)  ", it->first, it->second);
		// }
		// _printf("\n");

		const auto it1 = closing_above.upper_bound(close[bar_idx]);
		for (auto it = closing_above.begin(); it != it1; ) {
			// _printf("excercise: (%f, %bar_idx)\n", it->first, it->second);
			const int idx = it->second;
			end[idx] = std::min(end[idx], bar_idx);
			closing_below.erase(it_below[idx]);
			it = closing_above.erase(it_above[idx]);
		}

		const auto it2 = closing_below.lower_bound(close[bar_idx]);
		for (auto it = it2; it != closing_below.end(); ) {
			// _printf("excercise: (%f, %bar_idx)\n", it->first, it->second);
			const int idx = it->second;
			end[idx] = std::min(end[idx], bar_idx);
			closing_above.erase(it_above[idx]);
			it = closing_below.erase(it_below[idx]);
		}

		if (event[bar_idx]) {
			it_above[event_idx] = closing_above.insert({close[bar_idx]*(1 + unit_width[bar_idx]*pt_mul[bar_idx]), event_idx});
			it_below[event_idx] = closing_below.insert({close[bar_idx]*(1 - unit_width[bar_idx]*sl_mul[bar_idx]), event_idx});
			++event_idx;
		}

		// _printf("\n");
	}
}


template<bool enable_pt, bool enable_sl>
void triple_barrier_conditional(
	double* close,
	const int n_bars,
	int* vertical_barrier,
	double* unit_width,
	double* pt_mul,
	double* sl_mul,
	bool* event,
	int* end
) {
	int n_events = 0;
	for (int bar_idx = 0, event_idx = 0; bar_idx < n_bars; ++bar_idx) {
		if (event[bar_idx]) {
			end[event_idx] = std::min(n_bars-1, vertical_barrier[bar_idx]);
			++event_idx;
			++n_events;
		}
	}
	std::vector<std::multimap<const double, int>::iterator> it_below(n_events);
	std::vector<std::multimap<const double, int>::iterator> it_above(n_events);
	std::multimap<const double, int> pt_price;
	std::multimap<const double, int> sl_price;
	for (int bar_idx = 0, event_idx = 0; bar_idx < n_bars; ++bar_idx) {
		// _printf("bar #%bar_idx\n", bar_idx);
		// _printf("close: %f\n", close[bar_idx]);
		// _printf("pt_price ");
		// for (auto it = pt_price.begin(); it != pt_price.end(); ++it) {
		// 	_printf("(%f, %bar_idx)  ", it->first, it->second);
		// }
		// _printf("\nclosing_below ");
		// for (auto it = sl_price.begin(); it != sl_price.end(); ++it) {
		// 	_printf("(%f, %bar_idx)  ", it->first, it->second);
		// }
		// _printf("\n");

		if constexpr (enable_pt) {
			const auto it1 = pt_price.upper_bound(close[bar_idx]);
			for (auto it = pt_price.begin(); it != it1; ) {
				// _printf("excercise: (%f, %bar_idx)\n", it->first, it->second);
				const int idx = it->second;
				end[idx] = std::min(end[idx], bar_idx);
				if constexpr (enable_sl) {
					sl_price.erase(it_below[idx]);
				}
				it = pt_price.erase(it_above[idx]);
			}
		}

		if constexpr (enable_sl) {			
			const auto it2 = sl_price.lower_bound(close[bar_idx]);
			for (auto it = it2; it != sl_price.end(); ) {
				// _printf("excercise: (%f, %bar_idx)\n", it->first, it->second);
				const int idx = it->second;
				end[idx] = std::min(end[idx], bar_idx);
				if constexpr (enable_pt) {
					pt_price.erase(it_above[idx]);
				}
				it = sl_price.erase(it_below[idx]);
			}
		}

		if (event[bar_idx]) {
			if constexpr (enable_pt) {
				it_above[event_idx] = pt_price.insert({close[bar_idx]*(1 + unit_width[bar_idx]*pt_mul[bar_idx]), event_idx});
			}
			if constexpr (enable_sl) {
				it_below[event_idx] = sl_price.insert({close[bar_idx]*(1 - unit_width[bar_idx]*sl_mul[bar_idx]), event_idx});
			}
			++event_idx;
		}

		// _printf("\n");
	}
}


template<bool enable_pt, bool enable_sl>
void triple_barrier_conditional_threaded(
	double* close,
	const int n_bars,
	int* vertical_barrier,
	double* unit_width,
	double* pt_mul,
	double* sl_mul,
	bool* event,
	const int n_jobs,
	int* end
) {
	const int hw_concurrency = std::max<int>(1, std::thread::hardware_concurrency());
	const int n_threads = n_jobs < 1 ? hw_concurrency : n_jobs;
	const int bars_per_thread = n_bars / n_threads + (bool)(n_bars % n_threads);
//	printf("hw_concurrency = %i, bars_per_thread = %i\n", hw_concurrency, bars_per_thread);

	std::vector<int> fst_event(n_threads, std::numeric_limits<int>::max());

	int n_events = 0;
	for (int bar_idx = 0, event_idx = 0; bar_idx < n_bars; ++bar_idx) {
		if (event[bar_idx]) {
			fst_event[bar_idx / bars_per_thread] = std::min(fst_event[bar_idx / bars_per_thread], event_idx);
			end[event_idx] = std::min(n_bars-1, vertical_barrier[bar_idx]);
			++event_idx;
			++n_events;
		}
	}

	std::vector<std::multimap<const double, int>::iterator> it_below(n_events);
	std::vector<std::multimap<const double, int>::iterator> it_above(n_events);

	auto run_task = [&](const int bars_begin, const int bars_end, const int fst_event) -> void {
		std::multimap<const double, int> pt_price;
		std::multimap<const double, int> sl_price;
		for (int bar_idx = bars_begin, event_idx = fst_event; bar_idx < n_bars; ++bar_idx) {
			if constexpr (enable_pt) {
				const auto it1 = pt_price.upper_bound(close[bar_idx]);
				for (auto it = pt_price.begin(); it != it1; ) {
					const int idx = it->second;
					end[idx] = std::min(end[idx], bar_idx);
					if constexpr (enable_sl) {
						sl_price.erase(it_below[idx]);
					}
					it = pt_price.erase(it_above[idx]);
				}
			}

			if constexpr (enable_sl) {			
				const auto it2 = sl_price.lower_bound(close[bar_idx]);
				for (auto it = it2; it != sl_price.end(); ) {
					const int idx = it->second;
					end[idx] = std::min(end[idx], bar_idx);
					if constexpr (enable_pt) {
						pt_price.erase(it_above[idx]);
					}
					it = sl_price.erase(it_below[idx]);
				}
			}

			if (bar_idx < bars_end && event[bar_idx]) {
				if constexpr (enable_pt) {
					it_above[event_idx] = pt_price.insert({close[bar_idx]*(1 + unit_width[bar_idx]*pt_mul[bar_idx]), event_idx});
				}
				if constexpr (enable_sl) {
					it_below[event_idx] = sl_price.insert({close[bar_idx]*(1 - unit_width[bar_idx]*sl_mul[bar_idx]), event_idx});
				}
				++event_idx;
			}

			if (bar_idx >= bars_end && pt_price.size() + sl_price.size() == 0) {
				break;
			}
		}
	};

	std::vector<std::future<void>> tasks;
	for (int thread = 1; thread < n_threads; ++thread) {
		tasks.push_back(std::move(
				std::async(run_task, bars_per_thread * thread, std::min(n_bars, bars_per_thread * (thread + 1)), fst_event[thread])
		));
	}
	run_task(0, std::min(n_bars, bars_per_thread), fst_event[0]);
	for (int thread = 1; thread < n_threads; ++thread) {
		tasks[thread-1].get();
	}
}


extern "C" void triple_barrier(
	double* close,
	const int n_bars,
	int* vertical_barrier,
	double* unit_width,
	double* pt_mul,
	double* sl_mul,
	bool* event,
	int* end
) {
	if (pt_mul == 0 && sl_mul == 0) {
		triple_barrier_conditional<false, false>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, end);
	}
	if (pt_mul == 0 && sl_mul != 0) {
		triple_barrier_conditional<false, true>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, end);
	}
	if (pt_mul != 0 && sl_mul == 0) {
		triple_barrier_conditional<true, false>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, end);
	}
	if (pt_mul != 0 && sl_mul != 0) {
		triple_barrier_conditional<true, true>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, end);
	}
}


extern "C" void triple_barrier_threaded(
	double* close,
	const int n_bars,
	int* vertical_barrier,
	double* unit_width,
	double* pt_mul,
	double* sl_mul,
	bool* event,
	const int n_jobs,
	int* end
) {
	if (pt_mul == 0 && sl_mul == 0) {
		triple_barrier_conditional_threaded<false, false>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, n_jobs, end);
	}
	if (pt_mul == 0 && sl_mul != 0) {
		triple_barrier_conditional_threaded<false, true>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, n_jobs, end);
	}
	if (pt_mul != 0 && sl_mul == 0) {
		triple_barrier_conditional_threaded<true, false>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, n_jobs, end);
	}
	if (pt_mul != 0 && sl_mul != 0) {
		triple_barrier_conditional_threaded<true, true>(close, n_bars, vertical_barrier, unit_width, pt_mul, sl_mul, event, n_jobs, end);
	}
}

