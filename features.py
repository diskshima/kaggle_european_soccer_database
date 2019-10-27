# Generate FIFA data
fifa_data = get_fifa_data(match_data, player_stats_data, data_exists=False)

# Create features and labels.
bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
bk_cols_selected = ['B365', 'BW']
feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall=True)
inputs = feables.drop('match_api_id', axis=1)

# Explore data and create visualization.
labels = inputs.loc[:, 'label']
features = inputs.drop('label', axis=1)
features.head(5)
feature_details = explore_data(features, inputs, db_path)

# Split the data into train/calibrate and test data sets.
X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

# Split the training data into train and calibration data sets.
X_train, X_calibrate, y_train, y_calibrate = train_test_split(
    X_train_calibrate, y_train_calibrate, test_size=0.3, random_state=42,
    stratify=y_train_calibrate)

# Create cross validation data splits.
cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=5)
cv_sets.get_n_splits(X_train, y_train)
