# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation metrics for classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import scipy
import sklearn

from finetune import scorer


class SentenceLevelScorer(scorer.Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'] if 'label_ids' in results
                             else results['targets'])
    self._preds.append(results['predictions'])

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class F1Scorer(SentenceLevelScorer):
  """Computes F1 for classification tasks."""

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._positive_label = 1

  def _get_results(self):
    n_correct, n_predicted, n_gold = 0, 0, 0
    count, correct = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      # if pred == self._positive_label:
      if pred == self._positive_label:
          n_predicted += 1
      if y_true == self._positive_label:
        n_gold += 1
        if pred == y_true:
            n_correct += 1
      count += 1
      correct += (1 if y_true == pred else 0)
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * n_correct / n_predicted
      r = 100.0 * n_correct / n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('accuracy',  100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class ChemprotF1Scorer(SentenceLevelScorer):
    """Computes micro average F1 for chemprot RE task"""
    def __init__(self):
        super(ChemprotF1Scorer, self).__init__()
        self.label_2_int_mapper = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5':2, 'CPR:6': 3,
                                   'CPR:9': 4, 'False': 5}
        self.pos_labels= [0, 1, 2, 3, 4]
        
    def _get_results(self):
        p,r,f,_ = sklearn.metrics.precision_recall_fscore_support(y_pred=self._preds,
                                                                  y_true=self._true_labels,
                                                                  labels=self.pos_labels,
                                                                  average="micro")
        pr = p * 100
        recall = r * 100
        f1 = f * 100
        return [('precision', pr),
                ('recall', recall),
                ('f1', f1),]


class YesnoF1Scorer(SentenceLevelScorer):
  """Computes yesno F1 for classification tasks."""

  def __init__(self):
    super(YesnoF1Scorer, self).__init__()
    self._positive_label = 1

  def _get_results(self):
    n_no_correct, n_no_predicted, n_no_gold = 0, 0, 0
    n_yes_correct, n_yes_predicted, n_yes_gold = 0, 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      if y_true == self._positive_label:
        n_yes_gold += 1
        if pred == self._positive_label:
          n_yes_predicted += 1
          if pred == y_true:
            n_yes_correct += 1
        else:
          n_no_predicted += 1
      else:
        n_no_gold += 1
        if pred == 0:
          n_no_predicted += 1
          if pred == y_true:
              n_no_correct += 1
        else:
          n_yes_predicted += 1

    if n_yes_correct == 0:
      p_yes, r_yes, f1_yes = 0, 0, 0
    else:
      p_yes = 100.0 * n_yes_correct / n_yes_predicted
      r_yes = 100.0 * n_yes_correct / n_yes_gold
      f1_yes = 2 * p_yes * r_yes / (p_yes + r_yes)

    if n_no_correct == 0:
      p_no, r_no, f1_no = 0, 0, 0
    else:
      p_no = 100.0 * n_no_correct / n_no_predicted
      r_no = 100.0 * n_no_correct / n_no_gold
      f1_no = 2 * p_no * r_no / (p_no + r_no)
    return [
        ('precision_yes', p_yes),
        ('recall_yes', r_yes),
        ('f1_yes', f1_yes),
        ('precision_no', p_no),
        ('recall_no', r_no),
        ('f1_no', f1_no),
        ('loss', self.get_loss()),
    ]

class MCCScorer(SentenceLevelScorer):

  def _get_results(self):
    return [
        ('mcc', 100 * sklearn.metrics.matthews_corrcoef(
            self._true_labels, self._preds)),
        ('loss', self.get_loss()),
    ]


class RegressionScorer(SentenceLevelScorer):

  def _get_results(self):
    preds = np.array(self._preds).flatten()
    return [
        ('pearson', 100.0 * scipy.stats.pearsonr(
            self._true_labels, preds)[0]),
        ('spearman', 100.0 * scipy.stats.spearmanr(
            self._true_labels, preds)[0]),
        ('mse', np.mean(np.square(np.array(self._true_labels) - self._preds))),
        ('loss', self.get_loss()),
    ]
