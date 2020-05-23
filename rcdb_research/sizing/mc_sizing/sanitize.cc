#include "lib.h"
#include <cstdio>

int main() {
	{
		auto res = compute_bet_old(
			0.013,
			0.017,
			0.8,
			0.1,
			true,
			0.65,
			100,
			1e-4,
			300,
			5000
		);
		printf("old compounded = %f\n", res);
	}
	{
		auto res = compute_bet_old(
			// 0.013,
			// 0.017,
			// 0.8,
			// 0.1,
			// false,
			// 0.65,
			// 100,
			// 1e-4,
			// 300,
			// 5000

			0.013,
			0.017,
			0.70,
			0.50,
			false,
			0.7,
			100,
			1e-5,
			1000,
			5000
		);
		printf("old noncompounded = %f\n", res);
	}
	{
		auto res = compute_bet_old(
			0.013,
			0.017,
			0.8,
			0.1,
			true,
			0.65,
			100,
			1e-4,
			300,
			5000
		);
		printf("new compounded = %f\n", res);
	}
	{
		auto res = compute_bet_old(
			0.013,
			0.017,
			0.70,
			0.50,
			false,
			0.7,
			100,
			1e-5,
			1000,
			5000
		);
		printf("new noncompounded = %f\n", res);
	}
}