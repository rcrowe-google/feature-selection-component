import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe

from typing import Any, List
from enum import Enum


_selection_modes = {'percentile': SelectPercentile,
                        'k_best': SelectKBest,
                        'fpr': SelectFpr,
                        'fdr': SelectFdr,
                        'fwe': SelectFwe}

# our custom artifact will be for Feature scores
"""Custom Artifact type"""


class FeatureScores(tfx.types.artifact.Artifact):
    """Output artifact containing feature scores from the Feature Selection component"""
    TYPE_NAME = 'Feature Scores'
    PROPERTIES = {
        'span': standard_artifacts.SPAN_PROPERTY,
        'split_names': standard_artifacts.SPLIT_NAMES_PROPERTY,
    }

class FeaturePValues(tfx.types.artifact.Artifact):
    """Output artifact containing p-values scores from the Feature Selection component"""
    TYPE_NAME = 'Feature P-Values'
    PROPERTIES = {
        'span': standard_artifacts.SPAN_PROPERTY,
        'split_names': standard_artifacts.SPLIT_NAMES_PROPERTY,
    }

# our custom component will be feature selection


@component
def FeatureSelection(
    input_data: List,
    target_column: List,
    selector_func,
    score_func,
    num_param: int,
    column_names=[]
):
    """Feature Selection component

    Args:
        input_data: Input features in the form of list[list] where each inner list contains one record
        target_column: The target column which is inferred from `input_data`
        selector_func: feature selector type
        score_func: score function for feature selection. Example: chi2 etc.
        num_params: Parameter of the corresponding `selector_func`
        column_names: to generate artifact dictionaries containing scores
    """

    selector = _selection_modes[selector_func](score_func, k=num_param)
    selected_data = selector.fit_transform(input_data, target_column)

    selector_scores = selector.scores_
    selector_p_values = selector.pvalues_

    scores = dict(zip(column_names, selector_scores))
    pvalues = dict(zip(column_names, selector_p_values))

    return selected_data
