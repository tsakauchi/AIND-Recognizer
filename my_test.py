import numpy as np
from asl_data import AslDb
from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC
from my_recognizer import recognize
from asl_utils import show_errors


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3, verbose=False).select()
        model_dict[word]=model
    return model_dict


asl = AslDb()  # initializes the database

asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']

df_means = asl.df.groupby('speaker').mean()
asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])
asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])

df_std = asl.df.groupby('speaker').std()
asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])
asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']

asl.df['polar-rr'] = np.sqrt(np.power(asl.df['grnd-rx'], 2) + np.power(asl.df['grnd-ry'], 2))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(np.power(asl.df['grnd-lx'], 2) + np.power(asl.df['grnd-ly'], 2))
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['delta-rx'] = asl.df['grnd-rx'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['grnd-ry'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['grnd-lx'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['grnd-ly'].diff().fillna(0)
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

df_dmean = asl.df.groupby('speaker').mean()
df_dstd = asl.df.groupby('speaker').std()
asl.df['custom-rx'] = (asl.df['delta-rx'] - asl.df['speaker'].map(df_dmean['delta-rx'])) / asl.df['speaker'].map(df_dstd['delta-rx'])
asl.df['custom-ry'] = (asl.df['delta-ry'] - asl.df['speaker'].map(df_dmean['delta-ry'])) / asl.df['speaker'].map(df_dstd['delta-ry'])
asl.df['custom-lx'] = (asl.df['delta-lx'] - asl.df['speaker'].map(df_dmean['delta-lx'])) / asl.df['speaker'].map(df_dstd['delta-lx'])
asl.df['custom-ly'] = (asl.df['delta-ly'] - asl.df['speaker'].map(df_dmean['delta-ly'])) / asl.df['speaker'].map(df_dstd['delta-ly'])
features_custom = ['custom-rx', 'custom-ry', 'custom-lx', 'custom-ly']

features = dict()
features['features_norm'] = features_norm
features['features_polar'] = features_polar
features['features_delta'] = features_delta
features['features_custom'] = features_custom

model_selectors = dict()
model_selectors["SelectorCV"] = SelectorCV
model_selectors["SelectorBIC"] = SelectorBIC
model_selectors["SelectorDIC"] = SelectorDIC

# Recognize the test set and display the result with the show_errors method
for feature_name, feature in features.items():
    for model_selector_name, model_selector in model_selectors.items():
        models = train_all_words(feature, model_selector)
        test_set = asl.build_test(feature)
        probabilities, guesses = recognize(models, test_set)
        print("{} {}".format(feature_name, model_selector_name))
        show_errors(guesses, test_set)
