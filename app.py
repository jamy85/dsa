import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE SETUP & UI
# ==============================================================================
st.set_page_config(page_title="Spain Debt Sustainability Model", layout="wide")
st.title("Sovereign Debt Sustainability Dashboard")
st.markdown("""
Explore different macroeconomic shocks, fiscal adjustments, and the 'penalty of delay' using the parameters on the left.
*Navigate the different analyses using the tabs below.*
""")

with st.sidebar:
    st.header("1. Economic Baseline")
    CURRENT_DEBT_TO_GDP = st.number_input("Current Debt-to-GDP (%)", value=101.8, step=1.0)
    PRIMARY_DEFICIT_PCT = st.slider("Starting Primary Deficit (%)", min_value=-5.0, max_value=10.0, value=2.0, step=0.1)
    INFLATION_RATE = st.number_input("Inflation Rate (%)", value=2.0, step=0.5) / 100
    BASELINE_REAL_GROWTH = st.number_input("Baseline Real Growth (%)", value=1.5, step=0.1) / 100

    st.header("2. Scenario Inputs")
    st.markdown("*Enter comma-separated values to generate your 3x3 matrices.*")
    yields_input = st.text_input("Market Yield Scenarios (%)", "3, 5, 7")
    growth_input = st.text_input("Real Growth Scenarios (%)", "0, 1, 2")
    
    st.header("3. Debt Maturity Profile")
    st.markdown("*Adjust the near-term maturity wall. The remainder pushes to Year 6+.*")
    y1 = st.slider("Year 1 Maturity (%)", 0, 50, 14)
    y2 = st.slider("Year 2 Maturity (%)", 0, 50, 11)
    y3 = st.slider("Year 3 Maturity (%)", 0, 50, 10)
    y4 = st.slider("Year 4 Maturity (%)", 0, 50, 9)
    y5 = st.slider("Year 5 Maturity (%)", 0, 50, 9)
    
    remainder = 100 - (y1 + y2 + y3 + y4 + y5)
    st.info(f"**Year 6+ (Remainder): {remainder}%**")
    
    if remainder < 0:
        st.error("Profile exceeds 100%! Please reduce early year maturities.")
        st.stop()

# ==============================================================================
# 2. DATA PROCESSING & FUNCTIONS
# ==============================================================================
# Parse Scenario Inputs
try:
    scenarios_r = [float(x.strip())/100 for x in yields_input.split(',')]
    scenarios_g = [float(x.strip())/100 for x in growth_input.split(',')]
except:
    st.error("Please ensure scenarios are comma-separated numbers (e.g., 3, 5, 7)")
    st.stop()

CURRENT_GDP_NOMINAL = 1500.0     
CURRENT_DEBT_STOCK = (CURRENT_DEBT_TO_GDP / 100) * CURRENT_GDP_NOMINAL
AVG_COUPON_EXISTING = 0.024 

# Profile Generation
user_maturity_vector = [y1/100, y2/100, y3/100, y4/100, y5/100, remainder/100]
redemption_profile = np.array(user_maturity_vector) * CURRENT_DEBT_STOCK
extended_profile = list(redemption_profile[:5]) + [redemption_profile[5]/5] * 5
df_profile = pd.DataFrame({'Year_Index': np.arange(1, 11), 'Redemption': extended_profile})
df_profile['Cumulative_Redemption'] = df_profile['Redemption'].cumsum()
df_profile['Remaining_Principal_Start'] = CURRENT_DEBT_STOCK - df_profile['Cumulative_Redemption'].shift(1, fill_value=0)

# CORE MODELS (Cached for speed)
@st.cache_data
def run_forward_simulation(start_debt, start_gdp, r_scenarios, g_scenarios, p_deficit, profile):
    results = []
    for r in r_scenarios:
        for g in g_scenarios:
            current_gdp = start_gdp
            stock_new_debt = 0.0
            prev_debt = start_debt
            for t in range(1, 11):
                nom_growth = (1 + g) * (1 + INFLATION_RATE) - 1
                current_gdp *= (1 + nom_growth)
                
                row = profile[profile['Year_Index'] == t]
                principal_old = row['Remaining_Principal_Start'].values[0] if not row.empty else 0
                maturing_old = row['Redemption'].values[0] if not row.empty else 0
                
                interest_old = principal_old * AVG_COUPON_EXISTING
                interest_new = stock_new_debt * r
                total_interest = interest_old + interest_new
                
                gfn = maturing_old + (current_gdp * (p_deficit / 100)) + total_interest
                stock_new_debt += gfn
                
                remaining_old_end = max(0, principal_old - maturing_old)
                total_debt_stock = stock_new_debt + remaining_old_end
                
                debt_to_gdp = (total_debt_stock / current_gdp) * 100
                effective_rate = (total_interest / prev_debt) * 100 if prev_debt > 0 else 0
                
                results.append({
                    'Year': t, 'Market_Yield': r, 'Real_Growth': g,
                    'Scenario': f"Yield: {r*100:g}% | Growth: {g*100:g}%",
                    'Debt_to_GDP': debt_to_gdp, 'Interest_to_GDP': (total_interest / current_gdp) * 100,
                    'Effective_Rate': effective_rate
                })
                prev_debt = total_debt_stock
    return pd.DataFrame(results)

@st.cache_data
def run_required_growth_paths(start_debt, start_gdp, r_scenarios, targets_d, p_deficit, profile):
    records = []
    for r in r_scenarios:
        for target in targets_d:
            d_0 = start_debt / start_gdp
            target_d_dec = target / 100
            p_def = p_deficit / 100
            prev_D, prev_gdp, stock_new_debt = start_debt, start_gdp, 0.0
            
            for t in range(1, 11):
                d_t = d_0 + t * ((target_d_dec - d_0) / 3) if t <= 3 else target_d_dec
                row = profile[profile['Year_Index'] == t]
                principal_old = row['Remaining_Principal_Start'].values[0] if not row.empty else 0
                maturing_old = row['Redemption'].values[0] if not row.empty else 0
                
                total_interest = (principal_old * AVG_COUPON_EXISTING) + (stock_new_debt * r)
                gdp_t = (prev_D + total_interest) / (d_t - p_def)
                
                g_nom = (gdp_t / prev_gdp) - 1
                g_real = ((1 + g_nom) / (1 + INFLATION_RATE)) - 1
                
                D_t = d_t * gdp_t
                stock_new_debt = D_t - max(0, principal_old - maturing_old)
                prev_D, prev_gdp = D_t, gdp_t
                
                records.append({
                    'Year': t, 'Market_Yield': f"{r*100:g}%",
                    'Target_Debt_Level': f"{target:g}%", 'Req_Real_Growth': g_real * 100
                })
    return pd.DataFrame(records)

@st.cache_data
def run_delayed_fiscal_adjustment(start_debt, start_gdp, r_scenarios, g_scenarios, target_pct, p_def_start, start_years, profile):
    records = []
    for r in r_scenarios:
        for g in g_scenarios:
            for start_yr in start_years:
                target_d = target_pct / 100
                p_def = p_def_start / 100
                prev_D, prev_gdp, stock_new_debt = start_debt, start_gdp, 0.0
                d_at_start_minus_1 = start_debt / start_gdp 
                
                for t in range(1, 11):
                    nom_growth = (1 + g) * (1 + INFLATION_RATE) - 1
                    gdp_t = prev_gdp * (1 + nom_growth)
                    
                    row = profile[profile['Year_Index'] == t]
                    principal_old = row['Remaining_Principal_Start'].values[0] if not row.empty else 0
                    maturing_old = row['Redemption'].values[0] if not row.empty else 0
                    
                    total_interest = (principal_old * AVG_COUPON_EXISTING) + (stock_new_debt * r)

                    if t < start_yr:
                        gfn = maturing_old + (p_def * gdp_t) + total_interest
                        stock_new_debt += gfn
                        D_t = stock_new_debt + max(0, principal_old - maturing_old)
                        d_at_start_minus_1 = D_t / gdp_t
                        pb_t_req = -p_def 
                    else:
                        t_adj = t - start_yr + 1
                        d_t = d_at_start_minus_1 + t_adj * ((target_d - d_at_start_minus_1) / 3) if t_adj <= 3 else target_d
                        D_t = d_t * gdp_t
                        pb_t_req = -( (D_t - prev_D - total_interest) / gdp_t )
                        stock_new_debt = D_t - max(0, principal_old - maturing_old)

                    prev_D, prev_gdp = D_t, gdp_t
                    records.append({
                        'Year': t, 'Market_Yield': f"{r*100:g}%", 'Real_Growth': f"{g*100:g}%",
                        'Adjustment_Start_Year': f"Starts Year {start_yr}", 'Req_Primary_Balance': pb_t_req * 100
                    })
    return pd.DataFrame(records)

# Generate Data
all_results = run_forward_simulation(CURRENT_DEBT_STOCK, CURRENT_GDP_NOMINAL, scenarios_r, scenarios_g, PRIMARY_DEFICIT_PCT, df_profile)

# ==============================================================================
# 3. DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 1. Solvency & Liquidity", 
    "🚀 2. Required Growth to Target", 
    "⚖️ 3. Penalty of Delayed Action", 
    "📊 4. Year 3 vs 10 Comparison"
])

with tab1:
    st.subheader("Debt Trajectories & Stabilization Requirements")
    sns.set_style("whitegrid")
    
    # Line Charts
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=all_results, x='Year', y='Debt_to_GDP', hue='Scenario', linewidth=2, ax=axes1[0])
    axes1[0].set_title("Solvency: Debt-to-GDP", fontweight='bold')
    axes1[0].set_ylabel("% of GDP")
    axes1[0].get_legend().remove()
    
    sns.lineplot(data=all_results, x='Year', y='Interest_to_GDP', hue='Scenario', linewidth=2, ax=axes1[1])
    axes1[1].set_title("Liquidity: Interest Cost (% of GDP)", fontweight='bold')
    axes1[1].axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='Warning (4%)')
    axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    st.pyplot(fig1)
    
    # Heatmap
    st.markdown("---")
    st.markdown("**Long-Term Stabilization:** What primary balance is required at Year 10 to stop debt from rising further?")
    
    stab_df = all_results[all_results['Year'] == 10].copy()
    stab_df['Nom_Growth'] = (1 + stab_df['Real_Growth']) * (1 + INFLATION_RATE) - 1
    stab_df['Req_PB'] = ((stab_df['Effective_Rate']/100 - stab_df['Nom_Growth']) / (1 + stab_df['Nom_Growth'])) * stab_df['Debt_to_GDP']
    
    pivot_pb = stab_df.pivot(index="Market_Yield", columns="Real_Growth", values="Req_PB")
    pivot_debt = stab_df.pivot(index="Market_Yield", columns="Real_Growth", values="Debt_to_GDP")
    
    pivot_pb.index = [f"{x*100:g}%" for x in pivot_pb.index]
    pivot_pb.columns = [f"{x*100:g}%" for x in pivot_pb.columns]
    
    annot_matrix = pd.DataFrame(index=pivot_pb.index, columns=pivot_pb.columns)
    for r in pivot_pb.index:
        for c in pivot_pb.columns:
            pb_val = pivot_pb.loc[r, c]
            d_val = pivot_debt.loc[float(r.strip('%'))/100, float(c.strip('%'))/100]
            annot_matrix.loc[r, c] = f"{pb_val:.2f}%\n({int(round(d_val))}%)"

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_pb, annot=annot_matrix, fmt="", cmap="RdBu", center=0, ax=ax2, cbar_kws={'label': 'Primary Balance Required (%)'})
    ax2.set_ylabel("Market Yield")
    ax2.set_xlabel("Real GDP Growth")
    ax2.invert_yaxis()
    st.pyplot(fig2)


with tab2:
    st.subheader("Required Real GDP Growth to Stabilize Debt at Target")
    st.markdown("If we mandate a 3-year linear adjustment to a new debt target, how fast must the economy grow (assuming constant deficit)?")
    
    targets = [80.0, 90.0, 100.0, 110.0, 120.0]
    growth_paths = run_required_growth_paths(CURRENT_DEBT_STOCK, CURRENT_GDP_NOMINAL, scenarios_r, targets, PRIMARY_DEFICIT_PCT, df_profile)
    
    g_grid = sns.FacetGrid(growth_paths, col="Market_Yield", hue="Target_Debt_Level", height=5, aspect=1.1, palette="coolwarm_r")
    g_grid.map(sns.lineplot, "Year", "Req_Real_Growth", linewidth=3, marker="o")
    g_grid.set_axis_labels("Forecast Year", "Required Real Growth (%)")
    g_grid.set_titles(col_template="Market Rate: {col_name}", size=12, weight='bold')
    
    for ax in g_grid.axes.flat:
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xticks(range(1, 11))
    
    g_grid.add_legend(title="Target Debt-to-GDP", bbox_to_anchor=(1.05, 0.5))
    st.pyplot(g_grid.fig)


with tab3:
    st.subheader("The Fiscal Penalty of Delayed Action")
    st.markdown(f"If we delay adjustment, what level of austerity (Primary Surplus) is required to hit **90% Debt-to-GDP** in 3 years? *(Baseline assumes a constant {PRIMARY_DEFICIT_PCT}% Deficit until action is taken)*")
    
    target_debt_tab3 = st.slider("Select Target Debt-to-GDP for this analysis (%)", 80, 130, 90, step=5)
    
    fiscal_paths = run_delayed_fiscal_adjustment(CURRENT_DEBT_STOCK, CURRENT_GDP_NOMINAL, scenarios_r, scenarios_g, target_debt_tab3, PRIMARY_DEFICIT_PCT, [1, 3, 5], df_profile)
    
    f_grid = sns.FacetGrid(fiscal_paths, col="Real_Growth", row="Market_Yield", hue="Adjustment_Start_Year", height=3.5, aspect=1.4, palette="Set1")
    f_grid.map(sns.lineplot, "Year", "Req_Primary_Balance", linewidth=2.5, marker="o")
    f_grid.set_axis_labels("Forecast Year", "Req. Primary Balance (%)")
    f_grid.set_titles(row_template="Yield: {row_name}", col_template="Growth: {col_name}", size=11, weight='bold')
    
    for ax in f_grid.axes.flat:
        ax.axhline(0, color='black', linestyle='-')
        ax.axhline(-PRIMARY_DEFICIT_PCT, color='grey', linestyle='--', alpha=0.5)
        ax.set_xticks(range(1, 11))
        
    f_grid.add_legend(title="Timeline", bbox_to_anchor=(1.02, 0.5))
    st.pyplot(f_grid.fig)


with tab4:
    st.subheader("Cost of Delay: Year 3 vs Year 10 Requirements")
    st.markdown("How much higher does the stabilizing primary balance target move if adjustment is postponed?")
    
    df_y3 = all_results[all_results['Year'] == 3].copy()
    df_y3['Nom_Growth'] = (1 + df_y3['Real_Growth']) * (1 + INFLATION_RATE) - 1
    df_y3['Req_PB_Y3'] = ((df_y3['Effective_Rate']/100 - df_y3['Nom_Growth']) / (1 + df_y3['Nom_Growth'])) * df_y3['Debt_to_GDP']
    
    comp = pd.merge(df_y3[['Scenario', 'Debt_to_GDP', 'Req_PB_Y3']], stab_df[['Scenario', 'Debt_to_GDP', 'Req_PB']], on='Scenario')
    comp = comp.rename(columns={'Debt_to_GDP_x': 'Debt_Y3', 'Req_PB': 'Req_PB_Y10', 'Debt_to_GDP_y': 'Debt_Y10'})
    comp['Cost_of_Delay'] = comp['Req_PB_Y10'] - comp['Req_PB_Y3']
    comp = comp.sort_values('Req_PB_Y10', ascending=False).reset_index(drop=True)
    
    # Format for Streamlit dataframe
    comp_display = comp.copy()
    comp_display['Debt_Y3'] = comp_display['Debt_Y3'].map("{:.1f}%".format)
    comp_display['Debt_Y10'] = comp_display['Debt_Y10'].map("{:.1f}%".format)
    comp_display['Req_PB_Y3'] = comp_display['Req_PB_Y3'].map("{:.2f}%".format)
    comp_display['Req_PB_Y10'] = comp_display['Req_PB_Y10'].map("{:.2f}%".format)
    comp_display['Cost_of_Delay'] = comp_display['Cost_of_Delay'].map("{:+.2f}%".format)
    
    st.dataframe(comp_display, use_container_width=True)