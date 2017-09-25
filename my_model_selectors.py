import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        return self.hmm_model(num_states, self.X, self.lengths)

    def hmm_model(self, num_states, X, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def score_safe(self, model, X, lengths):
        # Hmmlearn problem: Rows of transmat_ must sum to 1.0
        # For some components quantity, there is not enough data.
        # Use try/except to catch these transmat errors inside the number of components for loop.
        # by letyrodri1
        # https: // discussions.udacity.com / t / hmmlearn - problem - rows - of - transmat - -must - sum - to - 1 - 0 / 249602
        try:
            return model.score(X, lengths)
        except:
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    L = likelihood of the fitted model
    p = number of parameters
        (NOTE: p = number of components^2 + ((number of components * number of features) * 2) - 1
    N = number of data points
    Evaluation condition: lower the better
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic_score = math.inf
        min_bic_score_model = None
        best_num_components = 0
        for num_components in range(self.min_n_components, self.max_n_components+1):

            model = self.base_model(num_components)
            if model is None:
                if self.verbose:
                    print("num_components={:3}, {}".format(
                        num_components,
                        "SKIPPED"))
                continue

            logL = self.score_safe(model, self.X, self.lengths)
            if logL is None:
                if self.verbose:
                    print("num_components={:3}, {}".format(
                        num_components,
                        "SKIPPED"))
                continue

            num_parameters = (num_components ** 2) + (2 * num_components * model.n_features) - 1

            logN = math.log10(sum(self.lengths))

            bic_score = -2 * logL + num_parameters * logN

            if self.verbose:
                print("num_components={:3}, logL={:10.1f}, bic={:10.1f}".format(
                    num_components,
                    logL,
                    bic_score))

            if min_bic_score > bic_score:
                min_bic_score = bic_score
                min_bic_score_model = model
                best_num_components = num_components

        if self.verbose:
            print("best_num_components={:3}, min_bic={:10.1f}".format(
                best_num_components,
                min_bic_score))

        return min_bic_score_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    DIC = (logL for target word) - (Average logL for all other known words)
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic_score = -math.inf
        max_dic_score_model = None
        best_num_components = 0

        for num_components in range(self.min_n_components, self.max_n_components+1):

            model = self.base_model(num_components)
            if model is None:
                if self.verbose:
                    print("num_components={:3}, {}".format(
                        num_components,
                        "SKIPPED"))
                continue

            logL = self.score_safe(model, self.X, self.lengths)
            if logL is None:
                if self.verbose:
                    print("num_components={:3}, {}".format(
                        num_components,
                        "SKIPPED"))
                continue

            other_logLs = []
            average_other_logL = 0
            for key, value in self.hwords.items():
                if key != self.this_word:
                    X, lengths = value
                    other_logL = self.score_safe(model, X, lengths)
                    if other_logL is None:
                        continue
                    other_logLs.append(other_logL)

            if len(other_logLs) > 0:
                average_other_logL = np.mean(other_logLs)

            dic_score = logL - average_other_logL

            if self.verbose:
                print("num_components={:3}, logL={:10.1f}, average_other_logL={:10.1f}, dic={:10.1f}".format(
                    num_components,
                    logL,
                    average_other_logL,
                    dic_score))

            if max_dic_score < dic_score:
                max_dic_score = dic_score
                max_dic_score_model = model
                best_num_components = num_components

        if self.verbose:
            print("best_num_components={:3}, max_dic={:10.1f}".format(
                best_num_components,
                max_dic_score))

        return max_dic_score_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if self.verbose:
            print("SelectorCV len_sequence={}".format(len(self.sequences)))

        # if number of sequences is less than 2, just return base model with constant number of components
        if len(self.sequences) < 2:
            return self.base_model(self.n_constant)

        split_method = KFold(min(3, len(self.sequences)))
        min_mean_logL = math.inf
        best_num_components = 0
        for num_components in range(self.min_n_components, self.max_n_components+1):
            logLs = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                trainX, trainLengths = combine_sequences(cv_train_idx, self.sequences)
                testX, testLengths = combine_sequences(cv_test_idx, self.sequences)
                model = self.hmm_model(num_components, trainX, trainLengths)

                if model is None:
                    if self.verbose:
                        print("num_components={:3}, len_seq={:3}, len_train={:3}, len_test={:3}, {}".format(
                            num_components,
                            len(self.sequences),
                            len(trainX),
                            len(testX),
                            "SKIPPED"))
                    continue

                logL = self.score_safe(model, testX, testLengths)

                if logL is None:
                    if self.verbose:
                        print("num_components={:3}, len_seq={:3}, len_train={:3}, len_test={:3}, {}".format(
                            num_components,
                            len(self.sequences),
                            len(trainX),
                            len(testX),
                            "SKIPPED"))
                    continue

                if self.verbose:
                    print("num_components={:3}, len_seq={:3}, len_train={:3}, len_test={:3}, logL={:10.1f}".format(
                        num_components,
                        len(self.sequences),
                        len(trainX),
                        len(testX),
                        logL))

                logLs.append(logL)

            if len(logLs) <= 0:
                if self.verbose:
                    print("num_components={:3}, {}".format(
                        num_components,
                        "SKIPPED"))
                continue

            mean_logL = np.mean(logLs)

            if self.verbose:
                print("num_components={:3}, mean_logL={:10.1f}".format(
                    num_components,
                    mean_logL))

            if min_mean_logL > mean_logL:
                min_mean_logL = mean_logL
                best_num_components = num_components

        if self.verbose:
            print("best_num_components={:3}, min_mean_logL={:10.1f}".format(
                best_num_components,
                min_mean_logL))

        return self.base_model(best_num_components)
