# Initialize models and parameters

# Initialize classifiers
RF_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
AB_clf = AdaBoostClassifier(n_estimators=200, random_state=2)
GNB_clf = GaussianNB()
KNN_clf = KNeighborsClassifier()
LOG_clf = linear_model.LogisticRegression(multi_class='ovr', solver='sag', class_weight='balanced')
clfs = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf]

# Grid search parameters
feature_len = features.shape[1]
scorer = make_scorer(accuracy_score)
dm_comp_count = np.arange(5, feature_len, np.around(feature_len / 5)).astype('int')
parameters_RF = {'clf__max_features': ['auto', 'log2'],
                 'dm_reduce__n_components': dm_comp_count}
parameters_AB = {'clf__learning_rate': np.linspace(0.5, 2, 5),
                 'dm_reduce__n_components': dm_comp_count}
parameters_GNB = {'dm_reduce__n_components': dm_comp_count}
parameters_KNN = {'clf__n_neighbors': [3, 5, 10],
                  'dm_reduce__n_components': dm_comp_count}
parameters_LOG = {'clf__C': np.logspace(1, 1000, 5),
                  'dm_reduce__n_components': dm_comp_count}
parameters = {clfs[0]: parameters_RF,
              clfs[1]: parameters_AB,
              clfs[2]: parameters_GNB,
              clfs[3]: parameters_KNN,
              clfs[4]: parameters_LOG}

# Dimensionality reduction
pca = PCA()
dm_reductions = [pca]
