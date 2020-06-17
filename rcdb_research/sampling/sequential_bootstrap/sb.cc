#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <set>

#include <iostream>
#include <cstdio>
#include <array>


struct ST {
	std::vector<double> segtree;
	std::vector<int> active;
	int n_spans;
	int max_idx;
	int max_tree_idx;

	ST(int max_idx, int n_spans)
		: segtree((max_idx+1)*4)
		, active(max_idx+1, 1)  // account for the "what if" case by adding +1 to the actual activity
		, n_spans(n_spans)
		, max_idx(max_idx)
		, max_tree_idx((1<<(31-__builtin_clz(max_idx+1)+1)) - 1)
		{}

	void refresh(int a, int b) {
		static_assert(sizeof(int) == 4);
		_refresh(a, b, 1, 0, max_tree_idx);
	}
	double get(int a, int b) {
		static_assert(sizeof(int) == 4);
		return _get(a, b, 1, 0, max_tree_idx);
	}

	void _refresh(int a, int b, int vptr, int left, int right) {
		// printf("_refresh a=%i, b=%i, vptr=%i, left=%i, right=%i\n", a, b, vptr, left, right);
		// if (left > right) { return; }
		if (left == right) { segtree[vptr] = 1./active[a]; return; }
		int mid = (left + right) / 2;
		if (a <= mid) {
			_refresh(a, std::min(mid, b), vptr*2, left, mid);
		}
		if (mid < b) {
			_refresh(std::max(mid+1, a), b, vptr*2+1, mid+1, right);
		}
		segtree[vptr] = segtree[vptr*2] + segtree[vptr*2+1];
	};

	double _get(int a, int b, int vptr, int left, int right) {
		// printf("_get a=%i, b=%i, vptr=%i, left=%i, right=%i\n", a, b, vptr, left, right);
		// if (left > right) { return 0; }
		if (left == a && right == b) { return segtree[vptr]; }
		int mid = (left + right) / 2;
		double res = 0;
		if (a <= mid) {
			res += _get(a, std::min(mid, b), vptr*2, left, mid);
		}
		if (mid < b) {
			res += _get(std::max(mid+1, a), b, vptr*2+1, mid+1, right);
		}
		return res;
	};
};

extern "C" void sequential_sample_st(
	const int (*const spans)[2], 
	const int n_spans,
	const int sample_size,
	const int seed,
	const int recalculate_every_n,
	int* result
) {
	// // test:
	// for (int i = 0; i < n_spans; ++i) {
	// 	printf("%i %i\n", spans[i][0], spans[i][1]);
	// }

	int max_idx = 1;

	// 1. Build coincidence
	std::vector<std::vector<int>> coincide(n_spans);
	for (int i = 0; i < n_spans; ++i) {
		max_idx = std::max({max_idx, spans[i][0], spans[i][1]});
		for (int j = i; j < n_spans; ++j) {
			if (spans[i][1] >= spans[j][0]) {
				coincide[i].push_back(j);
				coincide[j].push_back(i);
			} else {
				break;
			}
		}
	}

	// std::vector<double> segtree(n_spans*4);
	ST st(max_idx, n_spans);

	std::ranlux24_base engine(seed);
	// std::mt19937 engine(seed);
	std::uniform_int_distribution<int> intdistr(0, n_spans-1);
	std::uniform_real_distribution<double> realdistr(0, std::nextafter(1, std::numeric_limits<double>::max()));

	// std::vector<int> active(max_idx+1, 1);  // account for the "what if" case by adding +1 to the actual activity
	std::vector<double> uniqueness(n_spans, 1);
	st.refresh(0, max_idx);

	double max_uniqueness = 1;
	double sum_uniqueness = n_spans;

	// 2. Sample
	for (int i = 0; i < sample_size; ) {
		for (int k = 0; k < recalculate_every_n && i < sample_size; ++k, ++i) {
			int idx = 0;
			while (true) {
				idx = intdistr(engine);
				if (realdistr(engine)*max_uniqueness <= uniqueness[idx]) {
					break;
				}
			}

			result[i] = idx;
			const int a = spans[idx][0];
			const int b = spans[idx][1];
			for (int j = a; j < b+1; ++j) {
				st.active[j] += 1;
			}
			st.refresh(a, b);


			// 3. Update weights
			for (const int idx2 : coincide[idx]) {
				const int a = spans[idx2][0];
				const int b = spans[idx2][1];
				double u = st.get(a, b) / (b-a+1);
				sum_uniqueness -= std::exchange(uniqueness[idx2], u);
				sum_uniqueness += u;
				max_uniqueness = std::max(max_uniqueness, u);
			}
		}
	}
}


extern "C" void sequential_sample(
	const int (*const spans)[2], 
	const int n_spans,
	const int sample_size,
	const int seed,
	const int recalculate_every_n,
	int* result
) {
	// // test:
	// for (int i = 0; i < n_spans; ++i) {
	// 	printf("%i %i\n", spans[i][0], spans[i][1]);
	// }

	int max_idx = 1;

	// 1. Build coincidence
	std::vector<std::vector<int>> coincide(n_spans);
	for (int i = 0; i < n_spans; ++i) {
		max_idx = std::max({max_idx, spans[i][0], spans[i][1]});

		coincide[i].push_back(i);
		for (int j = i+1; j < n_spans; ++j) {
			if (spans[i][1] >= spans[j][0]) {
				coincide[i].push_back(j);
				coincide[j].push_back(i);
			} else {
				break;
			}
		}
	}

	std::ranlux24_base engine(seed);
	// std::mt19937 engine(seed);
	std::uniform_int_distribution<int> intdistr(0, n_spans-1);
	std::uniform_real_distribution<double> realdistr(0, std::nextafter(1, std::numeric_limits<double>::max()));

	std::vector<int> active(max_idx+1, 1);
	std::vector<double> uniqueness(n_spans, 1);

	double max_uniqueness = 1;
	double sum_uniqueness = n_spans;

	// 2. Sample
	for (int i = 0; i < sample_size; ) {
		for (int k = 0; k < recalculate_every_n && i < sample_size; ++k, ++i) {
			// const int idx = ddistr(engine);
			int idx = 0;
			while (true) {
				idx = intdistr(engine);
				if (realdistr(engine) <= uniqueness[idx]) {
					break;
				}
			}

			// printf("uniqueness ");
			// for (int i = 0; i < n_spans; ++i) {
			// 	printf("[%i: %f]", i, uniqueness[i]);
			// }
			// printf("\n");

			result[i] = idx;
			const int a = spans[idx][0];
			const int b = spans[idx][1];
			for (int j = a; j < b+1; ++j) {
				active[j] += 1;
			}


			// 3. Update weights
			for (const int idx2 : coincide[idx]) {
				const int a = spans[idx2][0];
				const int b = spans[idx2][1];
				double u = 0;
				for (int j = a; j < b+1; ++j) {
					u += 1./(active[j]);
				}


							// printf("prefixsum_before: ");
			// for (int i = 0; i < master_spansize+1; ++i) {
			// 	printf("[%i: %f]", i, prefixsum_before[i]);
			// }
			// printf("\n");
			// printf("prefixsum_after: ");
			// for (int i = 0; i < master_spansize+1; ++i) {
			// 	printf("[%i: %f]", i, prefixsum_after[i]);
			// }
			// printf("\n");

				// printf("master_a: %i  master_b: %i  a: %i  b: %i  overlap_skip: %i  overlap_take: %i\n", master_a, master_b, a, b, overlap_skip, overlap_take);

				sum_uniqueness -= uniqueness[idx2];
				uniqueness[idx2] = u / (b-a+1);
				max_uniqueness = std::max(max_uniqueness, uniqueness[idx2]);
				sum_uniqueness += uniqueness[idx2];
			}
		}
	}
}


extern "C" void sequential_sample_prefixsum(
	const int (*const spans)[2], 
	const int n_spans,
	const int sample_size,
	const int seed,
	const int recalculate_every_n,
	int* result
) {
	// // test:
	// for (int i = 0; i < n_spans; ++i) {
	// 	printf("%i %i\n", spans[i][0], spans[i][1]);
	// }

	int max_idx = 1;
	int max_spansize = 1;

	// 1. Build coincidence
	std::vector<std::vector<int>> coincide(n_spans);
	for (int i = 0; i < n_spans; ++i) {
		max_idx = std::max(max_idx, spans[i][1]);
		max_spansize = std::max(max_spansize, spans[i][1] - spans[i][0] + 1);

		coincide[i].push_back(i);
		for (int j = i+1; j < n_spans; ++j) {
			if (spans[i][1] >= spans[j][0]) {
				coincide[i].push_back(j);
				coincide[j].push_back(i);
			} else {
				break;
			}
		}
	}


	std::ranlux24_base engine(seed);
	// std::mt19937 engine(seed);

	std::uniform_int_distribution<int> intdistr(0, n_spans-1);
	std::uniform_real_distribution<double> realdistr(0, std::nextafter(1, std::numeric_limits<double>::max()));

	std::vector<int> active(max_idx+1, 1);
	std::vector<double> uniqueness(n_spans, 1);

	std::vector<std::array<double, 2>> prefixsum(max_spansize+1, {0});

	// 2. Sample
	for (int i = 0; i < sample_size; ++i) {
        int idx = 0;
        while (true) {
            idx = intdistr(engine);
            if (realdistr(engine)*1. <= uniqueness[idx]) {
                break;
            }
        }

        // printf("uniqueness ");
        // for (int i = 0; i < n_spans; ++i) {
        // 	printf("[%i: %f]", i, uniqueness[i]);
        // }
        // printf("\n");


        result[i] = idx;
        const int master_a = spans[idx][0];
        const int master_b = spans[idx][1];
        const int master_spansize = master_b - master_a + 1;

        for (int j = master_a, l = 1; j < master_b+1; ++j, ++l) {
            prefixsum[l][0] = prefixsum[l-1][0] + 1./active[j];
            active[j] += 1;
            prefixsum[l][1] = prefixsum[l-1][1] + 1./active[j];
        }

        // printf("active: ");
        // for (int i = 0; i < max_idx+1; ++i) {
        // 	printf("[%i: %i]", i, active[i]);
        // }
        // printf("\n");

        // printf("prefixsum_before: ");
        // for (int i = 0; i < master_spansize+1; ++i) {
        // 	printf("[%i: %f]", i, prefixsum[i][0]);
        // }
        // printf("\n");
        // printf("prefixsum_after: ");
        // for (int i = 0; i < master_spansize+1; ++i) {
        // 	printf("[%i: %f]", i, prefixsum[i][1]);
        // }
        // printf("\n");

        // 3. Update weights
        for (const int idx2 : coincide[idx]) {
            const int a = spans[idx2][0];
            const int b = spans[idx2][1];

            const int overlap_skip = std::max(0, a - master_a);
            const int overlap_take = master_spansize - std::max(0, master_b - b);

            // printf("master_a: %i  master_b: %i  a: %i  b: %i  overlap_skip: %i  overlap_take: %i\n", master_a, master_b, a, b, overlap_skip, overlap_take);

            const double div = 1./(b-a+1);

            const double subtract = (prefixsum[overlap_take][0] - prefixsum[overlap_skip][0]) * div;
            const double add = (prefixsum[overlap_take][1] - prefixsum[overlap_skip][1]) * div;

            uniqueness[idx2] += -subtract + add;
        }
	}
}


extern "C" void sequential_sample_prefixsum_optrng(
	const int (*const spans)[2], 
	const int n_spans,
	const int sample_size,
	const int seed,
	const int recalculate_every_n,
	int* result
) {
	// // test:
	// for (int i = 0; i < n_spans; ++i) {
	// 	printf("%i %i\n", spans[i][0], spans[i][1]);
	// }

	int max_idx = 1;
	int max_spansize = 1;

	// 1. Build coincidence
	std::vector<std::vector<int>> coincide(n_spans);
	for (int i = 0; i < n_spans; ++i) {
		max_idx = std::max(max_idx, spans[i][1]);
		max_spansize = std::max(max_spansize, spans[i][1] - spans[i][0] + 1);

		coincide[i].push_back(i);
		for (int j = i+1; j < n_spans; ++j) {
			if (spans[i][1] >= spans[j][0]) {
				coincide[i].push_back(j);
				coincide[j].push_back(i);
			} else {
				break;
			}
		}
	}


	std::ranlux24_base engine(seed);
	// std::mt19937 engine(seed);

	std::uniform_int_distribution<int> intdistr(0, n_spans-1);
	std::uniform_real_distribution<double> realdistr(0, std::nextafter(1, std::numeric_limits<double>::max()));

	std::vector<int> active(max_idx+1, 1);
	std::vector<double> uniqueness(n_spans, 1);

	std::vector<std::array<double, 2>> prefixsum(max_spansize+1, {0});

	std::multiset<double> maxuniq_ms(uniqueness.begin(), uniqueness.end());

	// 2. Sample
	for (int i = 0; i < sample_size; ) {
		for (int k = 0; k < recalculate_every_n && i < sample_size; ++k, ++i) {
			int idx = 0;
			while (true) {
				idx = intdistr(engine);
				// if (realdistr(engine)*1. <= uniqueness[idx]) {
				if (realdistr(engine) * *std::prev(maxuniq_ms.end()) <= uniqueness[idx]) {
					break;
				}
			}

			result[i] = idx;
			const int master_a = spans[idx][0];
			const int master_b = spans[idx][1];
			const int master_spansize = master_b - master_a + 1;

			for (int j = master_a, l = 1; j < master_b+1; ++j, ++l) {
				prefixsum[l][0] = prefixsum[l-1][0] + 1./active[j];
				active[j] += 1;
				prefixsum[l][1] = prefixsum[l-1][1] + 1./active[j];
			}

			// 3. Update weights
			for (const int idx2 : coincide[idx]) {
				const int a = spans[idx2][0];
				const int b = spans[idx2][1];

				const int overlap_skip = std::max(0, a - master_a);
				const int overlap_take = master_spansize - std::max(0, master_b - b);

				const double div = 1./(b-a+1);

				const double subtract = (prefixsum[overlap_take][0] - prefixsum[overlap_skip][0]) * div;
				const double add = (prefixsum[overlap_take][1] - prefixsum[overlap_skip][1]) * div;

				maxuniq_ms.erase(uniqueness[idx2]);
				uniqueness[idx2] += -subtract + add;
				maxuniq_ms.insert(uniqueness[idx2]);
			}
		}
	}
}



extern "C" void sequential_sample_orig(
	const int (*const spans)[2], 
	const int n_spans,
	const int sample_size,
	const int seed,
	const int recalculate_every_n,
	int* result
) {
	// // test:
	// for (int i = 0; i < n_spans; ++i) {
	// 	printf("%i %i\n", spans[i][0], spans[i][1]);
	// }

	int max_idx = 1;

	// 1. Build coincidence
	std::vector<std::vector<int>> coincide(n_spans);
	for (int i = 0; i < n_spans; ++i) {
		max_idx = std::max({max_idx, spans[i][0], spans[i][1]});

		for (int j = i; j < n_spans; ++j) {
			if (spans[i][1] >= spans[j][0]) {
				coincide[i].push_back(j);
				coincide[j].push_back(i);
			} else {
				break;
			}
		}
	}

	// std::ranlux24_base engine(seed);
	std::mt19937 engine(seed);

	std::vector<int> active(max_idx+1, 0);
	std::vector<float> uniqueness(n_spans, 1);

	// 2. Sample
	for (int i = 0; i < sample_size; ) {
		std::discrete_distribution<int> ddistr(uniqueness.begin(), uniqueness.end());
		for (int k = 0; k < recalculate_every_n && i < sample_size; ++k, ++i) {
			const int idx = ddistr(engine);
			result[i] = idx;
			const int a = spans[idx][0];
			const int b = spans[idx][1];
			for (int j = a; j < b+1; ++j) {
				active[j] += 1;
			}


			// 3. Update weights
			for (const int idx2 : coincide[idx]) {
				const int a = spans[idx2][0];
				const int b = spans[idx2][1];
				float u = 0;
				for (int j = a; j < b+1; ++j) {
					u += 1./(active[j] + 1);
				}
				uniqueness[idx2] = u / (b-a+1);
			}
		}
	}
}


/// Segtree outline:
/// 1) assign a range corresp to the span
/// 2) make a tree update over the range
/// 3) this can be considered a sum tree with assignment queries