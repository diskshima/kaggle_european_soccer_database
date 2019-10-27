# Run a grid search to find the best betting strategy based on the model.
percentile_grid = np.linspace(0, 1, 2)
probability_grid = np.linspace(0, 0.5, 2)
best_betting = optimize_betting(
    best_clf, best_dm_reduce, bk_cols_selected, bk_cols, match_data, fifa_data,
    5, 300, percentile_grid, probability_grid, verbose=True)
print('The best return of investment is: {}'.format(best_betting.results))
