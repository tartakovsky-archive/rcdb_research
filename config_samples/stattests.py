from commons.features import stattests

stattests_config = {
    'stattests': [
        dict(
            fn=stattests.kstest,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']},
        ),
        dict(
            fn=stattests.ks_2samp,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.epps_singleton_2samp,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.mannwhitneyu,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.wilcoxon,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.anderson_ksamp,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close', 'close_pct_change']}
        ),
        dict(
            fn=stattests.adfuller,
            pg={'window': [50, 200]},
            dm={'series': ['close']}
        ),
        dict(
            fn=stattests.chisquare,
            pg={'window': [50, 200], 'offset': [50, 100], 'bins': [5, 7]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.power_divergence,
            pg={'window': [50, 200], 'offset': [50, 100], 'bins': [5, 7]},
            dm={'series': ['close_pct_change']}
        ),
        dict(
            fn=stattests.runs_test,
            pg={'window': [50, 200], 'offset': [50, 100]},
            dm={'series': ['close_pct_change']}
        ),
    ]
}
