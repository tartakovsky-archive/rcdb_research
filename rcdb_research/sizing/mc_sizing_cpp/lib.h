using real_t = double;

// extern "C" real_t compute_b_via_mc_plus_cpp(
//     real_t avg_win,
//     real_t avg_loss,
//     const real_t minimum_wealth,
//     const real_t max_drawdown_risk,
//     real_t predicted_probability,
//     const real_t size_upper_bound,
//     const real_t xtol,
//     const int n_curves,
//     const int n_steps
// );

// extern "C" real_t compute_b_via_mc_plus_cpp_non_compounded(
//     real_t avg_win,
//     real_t avg_loss,
//     const real_t minimum_wealth,
//     const real_t max_drawdown_risk,
//     real_t predicted_probability,
//     const real_t size_upper_bound,
//     const real_t xtol,
//     const int n_curves,
//     const int n_steps
// );

extern "C" real_t compute_bet_old(
    const real_t avg_win,  // 0-based
    const real_t avg_loss,
    const real_t minimum_wealth,
    const real_t max_drawdown_risk,
    bool compounded,
    const real_t predicted_probability,
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
);

extern "C" real_t compute_bet(
    const real_t avg_win,  // 0-based
    const real_t avg_loss,
    const real_t minimum_wealth,
    const real_t max_drawdown_risk,
    bool compounded,
    const real_t predicted_probability,
    const real_t size_upper_bound,
    const real_t xtol,
    const int n_curves,
    const int n_steps
);

// extern "C" void toss_n_coins(real_t p, real_t n, int seed, real_t* out_buffer);
// extern "C" void run_n_simulations(...);
