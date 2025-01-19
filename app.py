import streamlit as st
import pandas as pd
import pickle

# Import your model and scenario function (assuming you've defined them elsewhere)
from scenario_utils import predict_scenario_before_after


X = pd.read_csv("data/mall_features.csv")

with open("trained_ridge.pkl", "rb") as f:
    ridge = pickle.load(f)


mall_names_df = pd.read_csv("mall_ids.csv")
# Suppose there's a column "mall_name" that has 22 rows, matching X's 0-21 index

mall_names_list = mall_names_df["mall_name"].tolist()

def main():
    st.title("Mall Tenant Mix Scenario Explorer")

    # Sliders
    count_apparel = st.slider("Change in Fashion Apparel Count", -10, 10, 0)
    count_fitness = st.slider("Change in Fitness Count", -10, 10, 0)
    count_Food_Beverage = st.slider("Change in Food & Beverage Services Count", -10, 10, 0)


    scenario_changes = {
        'count_Fashion apparel': count_apparel,
        'count_Fitness': count_fitness,
        'count_Food & Beverage Services': count_Food_Beverage
    }

    selected_mall_name = st.selectbox("Select a Mall", options=mall_names_list)
    mall_index = mall_names_list.index(selected_mall_name)

    # 4) Now retrieve the corresponding row in X
    default_row = X.iloc[mall_index]


    if st.button("Compute Scenario"):
        visits_before, visits_after = predict_scenario_before_after(
            ridge, default_row, scenario_changes, X.columns
        )


        st.write("Visits before the scenario change:", round(visits_before, 2))
        st.write("Visits after the scenario change:", round(visits_after, 2))
        st.write("Difference in visits:", round(visits_after - visits_before, 2))

if __name__ == "__main__":
    main()