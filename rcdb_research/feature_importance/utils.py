from operator import itemgetter
from itertools import groupby
from typing import List


def cluster_labels_to_clusters(labels: List[int], columns: List[str]) -> List[dict]:
    tuples = list(zip(labels, columns))
    groups = groupby(tuples, key=itemgetter(0))
    column_groups = [[v for k, v in g] for k, g in groups]

    clusters = [
        dict(
            name=f'{columns[0]}+{len(columns) - 1}',
            columns=columns
        )
        for columns in column_groups
    ]

    return clusters
