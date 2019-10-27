# Plot a confusion matrix of the best model and the bookkeeper predictions

best_clf = clfs[np.argmax(test_scores)]
best_dm_reduce = dm_reductions[np.argmax(test_scores)]
print('The best classifier is a {} with {}.'.format(
    best_clf.base_estimator.__class__.__name__, best_dm_reduce.__class__.__name__))
plot_confusion_matrix(y_test, X_test, best_clf, best_dm_reduce, path=db_path, normalize=True)

plot_bookkeeper_cf_matrix(match_data, bk_cols, db_path, verbose=True, normalize=True)
