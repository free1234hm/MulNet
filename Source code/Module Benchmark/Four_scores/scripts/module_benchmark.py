#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 11:55
from __future__ import division

import numpy as np

# for speed, the module comparison functions are implemented in Cython
import ebcubed
import jaccard
import json
from lib.modulecontainers import Modules
import os


def harmonic_mean(X):
    X = np.array(X)
    if np.any(X <= 0):
        return 0
    else:
        return len(X) / (np.sum(1 / np.array(X)))


class ModulesComparison():
    """
    Compares two sets of modules using several scores
    """

    def __init__(self, modulesA, modulesB, G):
        self.modulesA = modulesA
        self.modulesB = modulesB
        self.G = G

        self.membershipsA = self.modulesA.cal_membership(self.G).astype(np.uint8)
        self.membershipsB = self.modulesB.cal_membership(self.G).astype(np.uint8)
        #print(self.membershipsA)
        #print(self.membershipsB)

        if len(self.modulesB) > 0 and len(self.modulesA) > 0:
            self.jaccards = np.nan_to_num(
                jaccard.cal_similaritymatrix_jaccard(self.membershipsA.T.values, self.membershipsB.T.values))
        else:
            self.jaccards = np.zeros((1, 1))

    def score(self, baselines, scorenames=["rr", "rp", "consensus"]):
        """
        Scores two sets of modules

        """
        scores = {}

        # recovery and relevance
        if "rr" in scorenames:
            #print('rr...')
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                scores["recoveries"] = scores["relevances"] = np.zeros(1)
            else:
                scores["recoveries"] = self.jaccards.max(1)
                scores["relevances"] = self.jaccards.max(0)
            scores["recovery"] = scores["recoveries"].mean()
            scores["relevance"] = scores["relevances"].mean()
            scores["F1rr"] = harmonic_mean([scores["recovery"], scores["relevance"]])

        # recall and precision
        if "rp" in scorenames:
            #print('rp...')
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                #print('rp1.....')
                scores["recalls"] = scores["precisions"] = np.zeros(1)
                #print('rp2.....')
            else:
                #print('rp3.....')
                scores["recalls"], scores["precisions"] = ebcubed.cal_ebcubed(self.membershipsA.values,
                                                                              self.membershipsB.values,
                                                                              self.jaccards.T.astype(np.float64))
                #print('rp4.....')
            scores["recall"] = scores["recalls"].mean()
            scores["precision"] = scores["precisions"].mean()
            scores["F1rp"] = harmonic_mean([scores["recall"], scores["precision"]])

        # consensus score, uses the python munkres package
        '''
        if "consensus" in scorenames:
            print('consensus...')
            if (self.membershipsA.shape[1] == 0) or (self.membershipsB.shape[1] == 0):
                scores["consensus"] = 0
            else:
                cost_matrix = np.array(1 - self.jaccards, dtype=np.double).copy()
                indexes = Munkres(cost_matrix)
                consensus = (1 - cost_matrix[indexes]).sum() / max(self.jaccards.shape)
        '''
        if ("rr" in scorenames) and ("rp" in scorenames):
            scores["F1rprr"] = harmonic_mean(
                [scores["recall"], scores["precision"], scores["recovery"], scores["relevance"]])

        # compare with baseline
        if baselines is not None:
            for baseline_name, baseline in baselines.items():
                if "rr" in scorenames:
                    scores["F1rr_" + baseline_name] = harmonic_mean(
                        [(scores[scorename] / baseline[scorename]) for scorename in ["recovery", "relevance"]])
                if "rp" in scorenames:
                    scores["F1rp_" + baseline_name] = harmonic_mean(
                        [(scores[scorename] / baseline[scorename]) for scorename in ["recall", "precision"]])
                if ("rr" in scorenames) and ("rp" in scorenames):
                    scores["F1rprr_" + baseline_name] = harmonic_mean(
                        [(scores[scorename] / baseline[scorename]) for scorename in
                         ["recovery", "relevance", "recall", "precision"]])
                if "consensus" in scorenames:
                    scores["consensus" + baseline_name] = harmonic_mean(
                        [(scores[scorename] / baseline[scorename]) for scorename in ["consensus"]])

        return scores


def modevalscorer(modules, knownmodules, baselines=None):
    #print('all genes...')
    allgenes = sorted(list({g for module in knownmodules for g in module}))
    #print(len(allgenes), allgenes)
    #print('filteredmodules...')
    filteredmodules = modules.filter_retaingenes(allgenes).filter_size(5)
    #print('comp...')
    comp = ModulesComparison(filteredmodules, knownmodules, allgenes)
    settingscores = comp.score(baselines)

    return settingscores


def overlap_list(A, B):
    flat_A = {item for sublist in A for item in sublist}
    flat_B = {item for sublist in B for item in sublist}
    overlap = flat_A & flat_B
    filtered_B = [[item for item in sublist if item in overlap] for sublist in B]
    filtered_B = [sublist for sublist in filtered_B if len(sublist) > 1]

    module_B = Modules(filtered_B)
    return module_B


if __name__ == '__main__':

    module_folder = "..."  # Folder where module files are saved
    known_folder = "..."  # Folder where known module files are saved

    for root, dirs, files in os.walk(module_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            gene_modules = {}
            with open(file_path, 'r') as f:
                next(f)
                for line in f:
                    columns = line.strip().replace('"', "").split('\t')  # Split line by whitespace
                    if len(columns) >= 2:  # Ensure at least two columns exist
                        gene, module = columns[0], columns[1]
                        if module not in gene_modules:
                            gene_modules[module] = []
                        gene_modules[module].append(gene)

            gene_modules = Modules(list(gene_modules.values()))

            for root2, dirs2, files2 in os.walk(known_folder):
                for filename2 in files2:
                    file_path2 = os.path.join(root2, filename2)
                    known_modules = Modules(json.load(open(file_path2)))
                    known_modules = overlap_list(gene_modules, known_modules)
                    scores = modevalscorer(gene_modules, known_modules)
                    print(f"Module_file:{filename}\tKnown_modules:{filename2}\t"
                          f"recall:{scores['recall']}\tprecision:{scores['precision']}\trecovery:{scores['recovery']}\t"
                          f"relevance:{scores['relevance']}\tharmonic_mean:{scores['F1rprr']}")
