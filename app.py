import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE SETUP ---
st.set_page_config(page_title="Sovereign Debt Sustainability Model", layout="wide")
st.title("🇪🇸 Sovereign Debt Sustainability Dashboard")
st.markdown("Adjust the fiscal and economic parameters on the left to see how they impact debt trajectories.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Economic Baseline")
    CURRENT_DEBT_TO_GDP = st.number_input("Current Debt-to-GDP (%)", value=101.8, step=1.0)
    PRIMARY_DEFICIT_PCT = st.slider("Primary Deficit (%)", min_value=-5.0, max_value=10.0, value=2.0, step=0.1)
    
    st.header("2. Debt Maturity Profile")
    st.markdown("*Adjust the near-term maturity wall. The remainder will be pushed to Year 6+.*")
    
    y1 = st.slider("Year 1 Maturity (%)", 0, 50, 14)
    y2 = st.slider("Year 2 Maturity (%)", 0, 50, 11)
    y3 = st.slider("Year 3 Maturity (%)", 0, 50, 10)
    y4 = st.slider("Year 4 Maturity (%)", 0, 50, 9)
    y5 = st.slider("Year 5 Maturity (%)", 0, 50, 9)
    
    remainder = 100 - (y1 + y2 + y3 + y4 + y5)
    st.info(f"**Year 6+ (Remainder): {remainder}%**")
    
    # Normalize vector to decimal
    user_maturity_vector = [y1/100, y2/100, y3/100, y4/100, y5/100, remainder/100]

# --- MODEL LOGIC ---
# (This is a condensed version of your functions)
CURRENT_GDP_NOMINAL = 1500.0     
CURRENT_DEBT_STOCK = (CURRENT_DEBT_TO_GDP / 100) * CURRENT_GDP_NOMINAL
INFLATION_RATE = 0.02
AVG_COUPON_EXISTING = 0.024 

# Process Profile
redemption_profile = np.array(user_maturity_vector) * CURRENT_DEBT_STOCK
# We stretch the remainder evenly over years 6-10 for the model
extended_profile = list(redemption_profile[:5]) + [redemption_profile[5]/5] * 5
years_seq = np.arange(1, 11)

df_profile = pd.DataFrame({'Year_Index': years_seq, 'Redemption': extended_profile})
df_profile['Cumulative_Redemption'] = df_profile['Redemption'].cumsum()
df_profile['Remaining_Principal_Start'] = CURRENT_DEBT_STOCK - df_profile['Cumulative_Redemption'].shift(1, fill_value=0)

@st.cache_data # This tells Streamlit to run this fast!
def calculate_debt_path(start_debt, start_gdp, r_scenario, g_scenario, p_deficit_pct):
    records = []
    current_gdp = start_gdp
    stock_new_debt = 0.0
    for t in range(1, 11):
        nom_growth = (1 + g_scenario) * (1 + INFLATION_RATE) - 1
        current_gdp = current_gdp * (1 + nom_growth)
        
        row = df_profile[df_profile['Year_Index'] == t]
        principal_old = row['Remaining_Principal_Start'].values[0] if not row.empty else 0
        maturing_old = row['Redemption'].values[0] if not row.empty else 0
            
        interest_old = principal_old * AVG_COUPON_EXISTING
        interest_new = stock_new_debt * r_scenario
        total_interest = interest_old + interest_new
        
        gfn = maturing_old + (current_gdp * (p_deficit_pct / 100)) + total_interest
        stock_new_debt += gfn
        
        remaining_old_end = max(0, principal_old - maturing_old)
        total_debt_stock = stock_new_debt + remaining_old_end
        
        debt_to_gdp = (total_debt_stock / current_gdp) * 100
        
        records.append({
            'Year': t,
            'Scenario': f"Yield: {int(r_scenario*100)}% | Growth: {int(g_scenario*100)}%",
            'Debt_to_GDP': debt_to_gdp,
            'Interest_to_GDP': (total_interest / current_gdp) * 100
        })
    return pd.DataFrame(records)

# --- RUN SCENARIOS ---
scenarios_r = [0.03, 0.05, 0.07]
scenarios_g = [0.00, 0.01, 0.02]
results = []
for r in scenarios_r:
    for g in scenarios_g:
        results.append(calculate_debt_path(CURRENT_DEBT_STOCK, CURRENT_GDP_NOMINAL, r, g, PRIMARY_DEFICIT_PCT))
all_results = pd.concat(results)

# --- VISUALIZATION ---
st.header("Forecast Dashboard")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.lineplot(data=all_results, x='Year', y='Debt_to_GDP', hue='Scenario', linewidth=2, ax=axes[0])
axes[0].set_title("Solvency: Debt-to-GDP Trajectory", fontweight='bold')
axes[0].set_ylabel("% of GDP")
axes[0].get_legend().remove()

sns.lineplot(data=all_results, x='Year', y='Interest_to_GDP', hue='Scenario', linewidth=2, ax=axes[1])
axes[1].set_title("Liquidity: Interest Cost (% of GDP)", fontweight='bold')
axes[1].axhline(y=4.0, color='red', linestyle='--', label='Warning (4%)')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()

# VERY IMPORTANT: In Streamlit, you don't use plt.show(). You pass the figure to st.pyplot()
st.pyplot(fig)
