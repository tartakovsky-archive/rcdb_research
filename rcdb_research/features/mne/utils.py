def mne_doc_helper(mne_func):
    def inner(func):
        func.__doc__ = mne_func.__doc__\
            .replace("The signals.\n", "")\
            .replace(
                "data : ndarray, shape (n_channels, n_times)\n",
                "series : np.array\n\tInput data.\n\n    window : int\n\tWindow size\n\n"
            )
        return func
    return inner
