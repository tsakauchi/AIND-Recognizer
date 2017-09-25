import warnings
import math
from asl_data import SinglesData


def score_safe(model, X, lengths):
    # Hmmlearn problem: Rows of transmat_ must sum to 1.0
    # For some components quantity, there is not enough data.
    # Use try/except to catch these transmat errors inside the number of components for loop.
    # by letyrodri1
    # https: // discussions.udacity.com / t / hmmlearn - problem - rows - of - transmat - -must - sum - to - 1 - 0 / 249602
    try:
        return model.score(X, lengths)
    except:
        return None


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for X, lengths in test_set.get_all_Xlengths().values():
        probability = dict()
        max_probability = -math.inf
        max_probability_word = None
        for word, model in models.items():
            logL = score_safe(model, X, lengths)
            if logL is None:
                continue
            probability[word] = logL
            if logL > max_probability:
                max_probability = logL
                max_probability_word = word
        probabilities.append(probability)
        guesses.append(max_probability_word)

    return probabilities, guesses
