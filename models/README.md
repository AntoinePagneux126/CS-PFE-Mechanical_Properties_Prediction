# Models

---

This folder contains all our trained models.

* Random Forest
    - random_forest_A_1.pkl
    - random_forest_re02_1.pkl
    - random_forest_rm_1.pkl
    Each of these models are parametrized as following :
        RandomForestRegressor(   bootstrap=False,
                                        max_depth=110,
                                        max_features=8,
                                        min_samples_split=4,
                                        n_estimators=90,
                                        n_jobs=-1)
