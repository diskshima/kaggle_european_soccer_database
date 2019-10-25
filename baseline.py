# Train a baseline model (GBC)

clf = LOG_clf
clf.fit(X_train, y_train)
print('Score of {} for training set: {:.4f}.'.format(
    clf.__class__.__name__, accuracy_score(y_train, clf.predict(X_train))))
print('Score of {} for test set: {:.4f}.'.format(
    clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test))))

clfs, dm_reductions, train_scores, test_scores = find_best_classifier(
    clfs, dm_reductions, scorer, X_train, y_train, X_calibrate, y_calibrate,
    X_test, y_test, cv_sets, parameters, n_jobs)

plot_training_results(clfs, dm_reductions, np.array(train_scores),
                      np.array(test_scores), path=db_path)
