# Models

---

This folder contains all our trained models.

## Random Forest

* Random Forest #1
    - random_forest_A_1.pkl
    - random_forest_re02_1.pkl
    - random_forest_rm_1.pkl

    Each of these models are parametrized as following :
        RandomForestRegressor(  bootstrap=False,
                                max_depth=110,
                                max_features=8,
                                min_samples_split=4,
                                n_estimators=90,
                                n_jobs=-1)

* Random Forest #2
    - random_forest_A_2.pkl
        RandomForestRegressor(  bootstrap=False,
                                max_depth=20,
                                max_features=17,
                                min_samples_split=4,
                                n_estimators=110,
                                n_jobs=-1)
    - random_forest_re02_2.pkl
        RandomForestRegressor(  bootstrap=False,
                                max_depth=26,
                                max_features=17,
                                min_samples_split=4,
                                n_estimators=110,
                                n_jobs=-1)
    - random_forest_rm_2.pkl
        RandomForestRegressor(  bootstrap=False,
                                max_depth=24,
                                max_features=13,
                                min_samples_split=4,
                                n_estimators=80,
                                n_jobs=-1)
