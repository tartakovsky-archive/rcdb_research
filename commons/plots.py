import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import cufflinks
cufflinks.go_offline(connected=True)

def plot_grid(df, variables, n_rows, n_cols, kind='hist'):
    fig = plt.figure(figsize=(20, 20))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].plot(kind=kind, ax=ax)
        ax.set_title(i)
    fig.tight_layout()
    plt.show()

def plot_scores(scores, threshold=0, title='', xlabel='Observations', ylabel='Score'):
    scores[np.isnan(scores)] = threshold

    highest = scores.max()
    lowest = scores.min()
    mean = scores.mean()
    std = scores.std()

    size = scores.size
    index = list(range(1, size+1))

    def rgb_to_percents(l):
        return [c / 255 for c in l]

    blue = rgb_to_percents([116, 173, 209])
    orange = rgb_to_percents([244, 109, 67])

    bar_color = [blue if exp else orange for exp in (scores >= threshold)]

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 6)
    ax.set_frame_on(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlim(0, size + 10)
    plt.ylim(lowest - abs(lowest) * 0.05, highest * 1.02)
    plt.suptitle(title, fontsize=14)
    plt.title(
        f'mean: {mean*100:.2f}% | std: {std*100:.2f}% | low: {lowest*100:.2f}% | high: {highest*100:.2f}%',
        fontsize=12
    )
    plt.xlabel(xlabel, fontsize=12, labelpad=15)
    plt.ylabel(ylabel, fontsize=12, labelpad=15)

    plt.bar(index, scores - threshold, 1.2, bottom=[threshold] * len(scores), color=bar_color, label='score')
    plt.grid(color='lightgray', linestyle='-.', linewidth=0.5)
    plt.show()

def plot_cv_splits(cv, X, y=None):
    train_sets = pd.DataFrame(columns=['start', 'size', 'end'])
    test_sets = pd.DataFrame(columns=['start', 'size', 'end'])
    for index, (train, test) in enumerate(cv.split(X=X, y=y)):
        train_sets.loc[index, 'start'] = train[0]
        train_sets.loc[index, 'size'] = train[-1]-train[0]
        train_sets.loc[index, 'end'] = train[-1]
        test_sets.loc[index, 'start'] = test[0]
        test_sets.loc[index, 'size'] = test[-1]-test[0]
        test_sets.loc[index, 'end'] = test[-1]

    train_text = train_sets["start"].apply(lambda x: f'{x:.0f}...') + train_sets["end"].apply(lambda x: f'{x:.0f}')
    test_text = test_sets["start"].apply(lambda x: f'{x:.0f}...') + test_sets["end"].apply(lambda x: f'{x:.0f}')

    X.iplot(
        data=[
            {
                'name': 'Train set',
                'x': train_sets['size'],
                'y': train_sets.index,
                'base': train_sets['start'],
                'type': 'bar',
                'orientation': 'h',
                'text': train_text,
                'hoverinfo': "y+text"
            },
            {
                'name': 'Test set',
                'x': test_sets['size'],
                'y': test_sets.index,
                'base': test_sets['start'],
                'type': 'bar',
                'orientation': 'h',
                'text': test_text,
                'hoverinfo': "y+text"
            }
        ],
        layout={
            'barmode': 'stack',
            'xaxis': {
                'title': 'Observations',
                'automargin': True,
            },
            'yaxis': {
                'title': 'Split number',
                'automargin': True,
                'autorange': True,
            },
        }
    )
