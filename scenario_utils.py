# scenario_utils.py

import pandas as pd

def predict_scenario_before_after(model, mall_data, scenario_changes, feature_cols):
    """
    model: RandomForestRegressor or similar
    mall_data: a single row from your DataFrame (excluding 'mall_id' and the target)
    scenario_changes: dict of changes in categories
    feature_cols: the list of columns used in your model
    """
    # 1) Create a copy of the base row (before scenario change)
    base_scenario = mall_data.copy()

    # 2) Create a copy for the new scenario
    new_scenario = mall_data.copy()

    # 3) Apply the deltas in the scenario_changes dictionary
    for cat_col, delta in scenario_changes.items():
        # If cat_col wasn't in the row, default to 0
        new_scenario[cat_col] = new_scenario.get(cat_col, 0) + delta

    # 4) Convert each scenario to a 1-row DataFrame and reindex to match the model's feature columns
    base_df = pd.DataFrame([base_scenario]).reindex(columns=feature_cols, fill_value=0)
    new_df = pd.DataFrame([new_scenario]).reindex(columns=feature_cols, fill_value=0)

    # 5) Predict using the model
    visits_before = model.predict(base_df)[0]
    visits_after = model.predict(new_df)[0]

    return visits_before, visits_after
