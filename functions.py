import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from time import time
from pathlib import Path
import warnings

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline

def get_match_label(match):
    '''Derives a label for a given match.'''
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0, 'match_api_id'] = match['match_api_id']

    if home_goals > away_goals:
        label.loc[0, 'label'] = 'Win'
    if home_goals == away_goals:
        label.loc[0, 'label'] = 'Draw'
    if home_goals < away_goals:
        label.loc[0, 'label'] = 'Defeat'

    return label.loc[0]

def get_fifa_stats(match, player_stats):
    '''Aggregate FIFA stats for a given match.'''
    match_id = match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    for player in players:
        player_id = match[player]
        stats = player_stats[player_stats.player_api_id == player_id]
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]

        if np.isnan(player_id):
            # NOTE: Defaulting to the first item?
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace=True, drop=True)
            overall_rating = pd.Series(current_stats.loc[0, 'overall_rating'])

        name = '{}_overall_rating'.format(player)
        names.append(name)

        player_stats_new = pd.concat([player_stats_new, overall_rating], axis=1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace=True, drop=True)

    return player_stats_new.ix[0]

def get_fifa_data(matches, player_stats, path=None, data_exists=False):
    '''Gets FIFA data for all matches.'''
    if data_exists:
        fifa_data = pd.read_pickle(path)
    else:
        print('Collecting FIFA dta for each match...')
        start = time()
        fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis=1)
        end = time()
        print('FIFA data collected in {:.1f} minutes'.format((end - start)/60))

    return fifa_data

def get_overall_fifa_rankings(fifa, get_overall=False):
    '''Get overall FIFA rankings from FIFA data.'''
    temp_data = fifa

    if get_overall:
        data = temp_data.loc[:, fifa.columns.str.contains('overall_rating')]
        data.loc[:, 'match_api_id'] = temp_data.loc[:, 'match_api_id']
    else:
        cols = fifa.loc[:, fifa.columns.str.contains('date_stat')]
        temp_data = fifa.drop(cols.columns, axis=1)
        data = temp_data

    return data

def get_last_matches(matches, date, team, x=10):
    '''Get the last matches of a given team.'''
    team_matches = matches[
        (matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)
    ]
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    return last_matches

def get_last_matches_against_each_other(matches, date, home_team, away_team, x=10):
    '''Get the last x matches of two given teams.'''
    home_matches = matches[(matches['home_team_api_id'] == home_team) &
                           (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) &
                           (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    min_x = min(x, total_matches.shape[0])
    matches_before = total_matches[total_matches.date < date]
    matches_sorted = matches_before.sort_values(by='date', ascending=False)
    return matches_sorted.iloc[0:min_x, :]

def get_goals(matches, team):
    '''Get the goals of a specific team frm a set of matches.'''
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    return total_goals

def get_goals_conceded(matches, team):
    '''Get the goals conceded of a specific team from a set of matches.'''
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    return total_goals

def get_wins(matches, team):
    '''Get the number of wins of a specific team from a set of matches.'''
    home_wins = int(matches.home_team_goal[
        (matches.home_team_api_id == team) &
        (matches.home_team_goal > matches.away_team_goal)
    ].count())
    away_wins = int(matches.away_team_goal[
        (matches.away_team_api_id == team) &
        (matches.away_team_goal > matches.home_team_goal)
    ].count())
    total_wins = home_wins + away_wins
    return total_wins

def get_match_features(match, matches, x=10):
    '''Create match specific features for a given match.'''
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    matches_home_team = get_last_matches(matches, date, home_team, x)
    matches_away_team = get_last_matches(matches, date, away_team, x)

    last_matches_against = get_last_matches_against_each_other(
        matches, date, home_team, away_team, x # =3
    )

    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceded = get_goals_conceded(matches_home_team, home_team)
    away_goals_conceded = get_goals_conceded(matches_away_team, away_team)

    result = pd.DataFrame()
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceded
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceded
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)

    return result.loc[0]

def create_feables(matches, fifa, bookkeepers, get_overall=False, horizontal=True, x=10, verbose=True):
    '''Create and aggregate features and labels for all matches.'''

    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)

    if verbose:
        print('Generating match features...')

    start = time()

    match_stats = matches.apply(lambda x: get_match_features(x, matches, x=10), axis=1)

    dummies = pd.get_dummies(match_stats['league_id']).rename(columns=lambda x: 'League_' + str(x))
    match_stats = pd.concat([match_stats, dummies], axis=1)
    match_stats.drop(['league_id'], inplace=True, axis=1)
    end = time()

    if verbose:
        print('Match features generated in {:.1f} minutes'.format((end - start) / 60))

    if verbose:
        print('Generating match labels...')

    start = time()
    labels = matches.apply(get_match_label, axis=1)
    end = time()

    if verbose:
        print('Match labels generated in {:.1f} minutes'.format((end - start) / 60))

    if verbose:
        print('Generating bookkeeper data...')

    start = time()
    bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal=True)
    bk_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    end = time()
    if verbose:
        print('Bookkeeper data generated in {:.1f} minutes'.format((end - start) / 60))

    features = pd.merge(match_stats, fifa_stats, on='match_api_id', how='left')
    features = pd.merge(features, bk_data, on='match_api_id', how='left')
    feables = pd.merge(features, labels, on='match_api_id', how='left')

    feables.dropna(inplace=True)

    return feables

def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer,
                     jobs, use_grid_search=True, best_components=None, best_params=None):
    '''Fits a classifier to the training data.'''
    start = time()
    if use_grid_search:
        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)

        grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
        grid_obj.fit(X_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:
        estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)
        best_pipe = pipeline.fit(X_train, y_train)

    end = time()

    print('Trained {} in {:.1f} minutes'.format(clf.__class__.__name__, (end - start) / 60))

    return best_pipe

def predict_labels(clf, best_pipe, features, target):
    '''Make predictions using a fit classifier based on scorer.'''
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    end = time()

    print('Made predictions in {:.4f} seconds'.format(end - start))
    return accuracy_score(target.values, y_pred)

def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate,
                            X_test, y_test, cv_sets, params, scorer, jobs, use_grid_search=True, **kwargs):
    '''Train and predict using a classifier based on scorer.'''
    print('Training a {} with {}...'.format(clf.__class__.__name__,
                                            dm_reduction.__class__.__name__))
    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)

    print('Calibrating probabilities of classifier...')
    start = time()
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv='prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print('Calibrated {} in {:.1f} minutes'.format(clf.__class__.__name__, (end - start) / 60))

    print('Score of {} for training set: {:.4f}.'.format(
        clf.__class__.__name__, predict_labels(clf, best_pipe, X_train, y_train)))
    print('Score of {} for test set: {:.4f}.'.format(
        clf.__class__.__name__, predict_labels(clf, best_pipe, X_test, y_test)))

    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(
        clf, best_pipe, X_train, y_train), predict_labels(clf, best_pipe, X_test, y_test)

def convert_odds_to_prob(match_odds):
    '''Converts bookkeeper odds to probabilities.'''
    match_id = match_odds.loc[:, 'match_api_id']
    bookkeeper = match_odds.loc[:, 'bookkeeper']
    win_odd = match_odds.loc[:, 'Win']
    draw_odd = match_odds.loc[:, 'Draw']
    loss_odd = match_odds.loc[:, 'Defeat']

    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd

    total_prob = win_prob + draw_prob + loss_prob

    probs = pd.DataFrame()
    probs.loc[:, 'match_api_id'] = match_id
    probs.loc[:, 'bookkeeper'] = bookkeeper
    probs.loc[:, 'Win'] = win_prob / total_prob
    probs.loc[:, 'Draw'] = draw_prob / total_prob
    probs.loc[:, 'Defeat'] = loss_prob / total_prob

    return probs

# bookkeepers: Bookkeeper tag
def get_bookkeeper_data(matches, bookkeepers, horizontal=True):
    '''Aggregates bookkeeper data for all matches and bookkeepers'''
    bk_data = pd.DataFrame()

    for bookkeeper in bookkeepers:
        temp_data = matches.loc[:, matches.columns.str.contains(bookkeeper)]
        temp_data.loc[:, 'bookkeeper'] = str(bookkeeper)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

        cols = temp_data.columns.values
        cols[:3] = ['Win', 'Draw', 'Defeat']
        temp_data.columns = cols
        temp_data.loc[:, 'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:, 'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:, 'Defeat'] = pd.to_numeric(temp_data['Defeat'])

        if horizontal:
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis=1, inplace=True)
            temp_data.drop('bookkeeper', axis=1, inplace=True)

            win_name = bookkeeper + '_Win'
            draw_name = bookkeeper + '_Draw'
            defeat_name = bookkeeper + '_Defeat'
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]

            bk_data = pd.concat([bk_data, temp_data], axis=1)
        else:
            bk_data = bk_data.append(temp_data, ignore_index=True)

    if horizontal:
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

    return bk_data

def get_bookkeeper_probs(matches, bookkeepers, horizontal=False):
    '''Get bookkeeper data and convert to probabilities for vertical aggregation.'''
    data = get_bookkeeper_data(matches, bookkeepers, horizontal=False)
    probs = convert_odds_to_prob(data)
    return probs

def plot_confusion_matrix(y_test, X_test, clf, dim_reduce, path, cmap=plt.cm.Blues, normalize=False):
    '''Plot confusion matrix for given classifier and data.'''
    labels = ['Win', 'Draw', 'Defeat']
    cm = confusion_matrix(y_test, clf.predict(dim_reduce.transform(X_test)), labels)

    if normalize:
        cm = cm.astype('float') / cm.sum()

    sns.set_style('whitegrid', {'axes.grid': False})
    fig = plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = 'Confusion matrix of a {} with {}'.format(
        best_clf.base_estimator.__class__.__name__,
        best_dm_reduce.__class__.__name__
    )
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    y_pred = clf.predict(dim_reduce.transform(X_test))
    print(classification_report(y_test, y_pred))

def compare_probabilities(clf, dim_reduce, bk, bookkeepers, matches, fifa_data, verbose=False):
    '''Map bookkeeper and model probabilities.'''
    feables = create_feables(matches, fifa_data, bk, get_overall=True, verbose=False)

    match_ids = list(feables['match_api_id'])
    matches = matches[matches['match_api_id'].isin(match_ids)]

    if verbose:
        print('Obtaining bookkeeper probabilities...')

    bookkeeper_probs = get_bookkeeper_probs(matches, bookkeepers)
    bookkeeper_probs.reset_index(inplace=True, drop=True)

    inputs = feables.drop('match_api_id', axis=1)
    labels = inputs.loc[:, 'label']
    features = inputs.drop('label', axis=1)

    if verbose:
        print('Predicting probabitlies based on model...')
    model_probs = pd.DataFrame()
    label_table = pd.Series()
    temp_probs = pd.DataFrame(
        clf.predict_proba(dim_reduce.transform(features)),
        columns=['win_prob', 'draw_prob', 'defeat_prob'])

    for bookkeeper in bookkeepers:
        model_probs = model_probs.append(temp_probs, ignore_index=True)
        label_table = label_table.append(labels)

    model_probs.reset_index(inplace=True, drop=True)
    label_table.reset_index(inplace=True, drop=True)
    bookkeeper_probs['win_prob'] = model_probs['win_prob']
    bookkeeper_probs['draw_prob'] = model_probs['draw_prob']
    bookkeeper_probs['defeat_prob'] = model_probs['defeat_prob']
    bookkeeper_probs['label'] = label_table

    wins = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Win', 'win_prob', 'label']]
    wins.loc[:, 'bet'] = 'Win'
    wins = wins.rename(columns={'Win': 'bookkeeper_prob',
                                'win_prob': 'model_prob'})

    draws = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Draw', 'draw_prob', 'label']]
    draws.loc[:, 'bet'] = 'draw'
    draws = draws.rename(columns={'Draw': 'bookkeeper_prob',
                                  'draw_prob': 'model_prob'})

    defeats = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Defeat', 'defeat_prob', 'label']]
    defeats.loc[:, 'bet'] = 'defeat'
    defeats = defeats.rename(columns={'Defeat': 'bookkeeper_prob',
                                      'defeat_prob': 'model_prob'})

    total = pd.concat([wins, draws, defeats])

    return total

def find_good_bets(clf, dim_reduce, bk, bookkeepers, matches, fifa_data,
                   percentile, prob_cap, verbose=False):
    '''Find good bets for a given classifier and matches.'''
    probs = compare_probabilities(clf, dim_reduce, bk, bookkeepers, matches,
                                  fifa_data, verbose)
    probs.loc[:, 'prob_difference'] = probs.loc[:, 'model_prob'] - probs.loc[:, 'bookkeeper_prob']

    values = probs['prob_difference']
    values = values.sort_values(ascending=False)
    values.reset_index(inplace=True, drop=True)

    if verbose:
        print('Selecting attractive bets...')

    relevant_choices = probs[(probs.prob_difference > 0) & (probs.model_prob > prob_cap) & (probs.bet != 'Draw')]

    top_percent = 1 - percentile
    choices = relevant_choices[relevant_choices.prob_difference >= relevant_choices.prob_difference.quantile(top_percent)]
    choices.reset_index(inplace=True, drop=True)

    return choices

def get_reward(choice, matches):
    '''Get the reward of a given bet.'''
    match = matches[matches.home_team_api_id == choice.match_api_id]
    bet_data = match.loc[:, (match.columns.str.contains(choice.bookkeeper))]
    cols = bet_data.columns.values
    cols[:3] = ['win', 'draw', 'defeat']
    bet_data.columns = cols

    if choice.bet == 'Win':
        bet_quota = bet_data.win.values
    elif choice.bet == 'Draw':
        bet_quota = bet_data.draw.values
    elif choice.bet == 'Defeat':
        bet_quota = bet_data.defeat.values
    else:
        print('Error')

    if choice.bet == choice.label:
        reward = bet_quota
    else:
        reward = 0

    return reward

def execute_bets(bet_choices, matches, verbose=False):
    '''Get rewards for all bets.'''
    if verbose:
        print('Obtaining reward for chosen bets...')
    total_reward = 0
    total_invested = 0

    loops = np.arange(0, bet_choices.shape[0])
    for i in loops:
        reward = get_reward(bet_choices.iloc[i, :], matches)
        total_reward = total_reward + reward
        total_invested += 1

    investment_return = float(total_reward / total_invested) - 1
    return investment_return

def explore_data(features, inputs, path):
    '''Explore data by plotting KDE graphs.'''
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=-1, left=0.025, top=2, right=0.975)

    i = 1
    for col in features.columns:
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=0.5, rc={'lines.linewidth': 1})
        plt.subplot(7, 7, i)
        j = i - 1

        sns.distplot(inputs[inputs['label'] == 'Win'].iloc[:, j], hist=False, label='Win')
        sns.distplot(inputs[inputs['label'] == 'Draw'].iloc[:, j], hist=False, label='Draw')
        sns.distplot(inputs[inputs['label'] == 'Defeat'].iloc[:, j], hist=False, label='Defeat')
        plt.legend()
        i = i + 1

    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * 1.2, DefaultSize[1] * 1.2))
    plt.show()

    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    feature_details = features.describe().transpose()

    return feature_details

def find_best_classifier(classifiers, dm_reductions, scorer, X_t, y_t, X_c, y_c,
                         X_v, y_v, cv_sets, params, jobs):
    '''Tune all classifier and dimensionality reduction combinations to find best classifier.'''
    clfs_return = []
    dm_reduce_return = []
    train_scores = []
    test_scores = []

    for dm in dm_reductions:
        for clf in clfs:
            clf, dm_reduce, train_score, test_score = train_calibrate_predict(
                clf=clf, dm_reduction=dm, X_train=X_t, y_train=y_t, X_calibrate=X_c,
                y_calibrate=y_c, X_test=X_v, y_test=y_v, cv_sets=cv_sets,
                params=params[clf], scorer=scorer, jobs=jobs, use_grid_search=True)

            clfs_return.append(clf)
            dm_reduce_return.append(dm_reduce)
            train_scores.append(train_score)
            test_scores.append(test_score)

        return clfs_return, dm_reduce_return, train_scores, test_scores

def plot_training_results(clfs, dm_reductions, train_scores, test_scores, path):
    '''Plot results of classifier training.'''
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1, rc={'lines.linewidth': 1})
    ax = plt.subplot(111)
    w = 0.5
    x = np.arange(len(train_scores))
    ax.set_yticks(x + w)
    ax.legend ((train_scores[0], test_scores[0]), ('Train Scores', 'Test Socores'))
    names = []

    for i in range(0, len(clfs)):
        clf = clfs[i]
        clf_name = clf.base_estimator.__class__.__name__
        dm = dm_reductions[i]
        dm_name = dm.__class__.__name__

        name = '{} with {}'.format(clf_name, dm_name)
        names.append(name)

    ax.set_yticklabels((names))
    plt.xlim(0.5, 0.55)
    plt.barh(x, test_scores, color='b', alpha=0.6)
    plt.title('Test Data Accuracy Scores')
    fig = plt.figure(1)
    plt.show()

def optimize_betting(best_clf, best_dm_reduce, bk_cols_selected, bk_cols,
                     match_data, fifa_data, n_samples, sample_size, parameter_1_grid,
                     parameter_2_grid, verbose=False):
    '''True parameters of bet selection algorithm.'''
    samples = []
    for i in range(0, n_samples):
        sample = match_data.sample(n=sample_size, random_state=42)
        samples.append(sample)

    results = pd.DataFrame(columns=['parameter_1', 'parameter_2', 'results'])
    row = 0

    for i in parameter_1_grid:
        for j in parameter_2_grid:
            profits = []
            for sample in samples:
                choices = find_good_bets(best_clf, best_dm_reduce, bk_cols_selected,
                                         bk_cols, sample, fifa_data, i, j)
                profit = execute_bets(choices, match_data)
                profits.append(profit)
            result = np.mean(np.array(profits))
            results.loc[row, 'results'] = result
            results.loc[row, 'parameter_1'] = i
            results.loc[row, 'parameter_2'] = j
            row = row + 1
            if verbose:
                print('Simulated parameter combination: {}'.format(row))

    best_result = results.ix[results['results'].idxmax()]
    return best_result

def plot_bookkeeper_cf_matrix(matches, bookkeepers, path, verbose=False, normalize=True):
    '''Plot confusion matrix of bookkeeper predictions.'''
    if verbose:
        print('Obtaining labels...')

    y_test_temp = matches.apply(get_match_label, axis=1)

    if verbose:
        print('Obtaining bookkeeper probabilities...')

    bookkeeper_probs = get_bookkeeper_probs(matches, bookkeepers)
    bookkeeper_probs.reset_index(inplace=True, drop=True)
    bookkeeper_probs.dropna(inplace=True)

    if verbose:
        print('Obtaining bookkeeper labels...')

    y_pred_temp = pd.DataFrame()
    y_pred_temp.loc[:, 'bk_label'] = bookkeeper_probs[['Win', 'Draw', 'Defeat']].idxmax(axis=1)
    y_pred_temp.loc[:, 'match_api_id'] = bookkeeper_probs.loc[:, 'match_api_id']

    if verbose:
        print('Plotting confusion matrix...')

    results = pd.merge(y_pred_temp, y_test_temp, on='match_api_id', how='left')
    y_test = results.loc[:, 'label']
    y_pred = results.loc[:, 'bk_label']

    labels = ['Win', 'Draw', 'Defeat']
    cm = confusion_matrix(y_test, y_pred, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum()

    sns.set_style('whitegrid', {'axes.grid': False})
    fig = plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = 'Confusion matrix of Bookkeeper predictions!'
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test, y_pred))
    print('Bookkeeper score for test set: {:.4f}.'.format(accuracy_score(y_test, y_pred)))
