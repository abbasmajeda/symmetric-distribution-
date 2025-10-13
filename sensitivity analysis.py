#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:11:16 2025

@author: majeed
"""

from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------- CONFIG / DEFAULTS ----------
a, b = 0.25, 0.50 # Ångström–Prescott
PR0 = 0.75 # baseline performance ratio
gamma = -0.004 # per °C temp coefficient
NOCT = 45.0 # °C
G_sc = 1367.0 # solar constant W/m^2
days_in_month = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

sunshine_hours = [7.2, 7.8, 8.2, 8.2, 6.8, 4.5, 4, 4.7, 4.1, 4.6, 5.7, 6.5] 
temperature =[8.71757583618164, 10.1511489868164, 14.5156967163086, 15.9752258300781, 18.5500701904297, 19.8355804443359, 20.0574478149414, 20.761784362793, 18.8176284790039, 15.5126831054688, 12.2841522216797, 9.10032119750977]

wh_per_hour = 0.58
Temp_coef = -0.004
Tref = 25
 
def deg2rad(deg):
    return deg * np.pi / 180.0

def midmonth_day_of_year(month):
    # approximate day-of-year for mid-month (15th) in a non-leap year
    mdays_cum = np.array([0,31,59,90,120,151,181,212,243,273,304,334])
    return int(mdays_cum[month-1] + 15)

def extraterrestrial_daily_radiation_kwh(lat_deg, day_of_year):
    phi = deg2rad(lat_deg)
    n = day_of_year
    dr = 1.0 + 0.033 * np.cos(2*np.pi * n / 365.0)
    delta = 0.409 * np.sin(2*np.pi * n / 365.0 - 1.39)
    cos_ws = -np.tan(phi) * np.tan(delta)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)
    ws = np.arccos(cos_ws)
    S0 = (24.0/np.pi) * ws
    H0_wh = (24.0/np.pi) * G_sc * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    H0 = H0_wh / 1000.0
    return H0, S0

def compute_monthly(month):

    n = midmonth_day_of_year(month)
    H0, S0 = extraterrestrial_daily_radiation_kwh(25, n)
    S = sunshine_hours[month-1]
    if S0 <= 0:
        H = 0.0
    else:
        H = H0 * (a + b * (S / S0))
    Gavg = (H * 1000.0) / max(S, 0.1) # W/m2
    Tcell = temperature[month-1] + (NOCT - 20.0) / 800.0 * Gavg
    fT = 1.0 + gamma * (Tcell - Tref)
    PRtemp = PR0 * fT
    e_month = H * PRtemp  # kWh per kW per day
    return e_month
    

 
def run_microgrid_model(battery_capacity, solar_panel_count):

    model = ConcreteModel()
    
    #Sets    
    t_intervals = [(month, period) for month in range(1, 13) for period in range(1, 5)] 
    model.T = Set(initialize=t_intervals, dimen=2)
    
    #Parameters
    solar_generation = {} 
    cost_panels = solar_panel_count * 80 # example: 100 panels x $100
    cost_battery = battery_capacity * 80# battery bank cost
    cost_inverter = 5000
    cost_stc_switch = 4000
    cost_container = 8 * battery_capacity
    battery_cycle_cost = cost_battery / 6000
    # Expected component lifetimes (in years)
    panel_lifetime = 25
    inverter_lifetime = 5
    system_lifetime = 25 # analysis period
    df_load = pd.read_csv("load_profile_long.csv")
    load = df_load.set_index(['Month','Period_num'])['Load'].to_dict()

    for month in range(1, 13): 

        solar_generation[(month, 1)] = 0.5 * compute_monthly(month) * solar_panel_count * wh_per_hour
        solar_generation[(month, 2)] = 0.5 * compute_monthly(month) * solar_panel_count * wh_per_hour
        solar_generation[(month, 3)] = 0 # No solar generation at night
        solar_generation[(month, 4)] = 0
        
    model.SolarGeneration = Param(model.T, initialize=solar_generation)
    
    #load_profile = {(month, period): 400 for (month, period) in model.T } # Dummy load in kWh 
    
    model.Load = Param(model.T, initialize=load)
    
    #Battery parameters
    
    max_charge = (battery_capacity) # in kWh per interval 
    max_discharge = (battery_capacity)  # in kWh per interval
    
    model.ChargeAllowed = Var(model.T, domain=Binary)
    
    M = battery_capacity # Big M value, or max possible charge
    
    #Variables
    model.EnergyBought = Var(model.T, within=NonNegativeReals) 
    model.EnergySold = Var(model.T, within=NonNegativeReals) 
    model.BatterySOC = Var(model.T, bounds=(0, battery_capacity)) 
    model.BatteryCharge = Var(model.T, within=NonNegativeReals) 
    model.BatteryDischarge = Var(model.T, within=NonNegativeReals)
    model.IsCharging = Var(model.T, domain=Binary)
    
    
    #Dynamic prices (to be optimized)
    
    model.BuyPrice = Var(model.T, bounds=(0.1, 0.2)) 
    model.SellPrice = Var(model.T, bounds=(0.05, 0.15))
    
    
    #Initial SOC
    
    
    big_M = battery_capacity
    model.ChargeControl = Constraint(model.T, rule=lambda m, month, period: m.BatteryCharge[month, period] <= big_M * m.IsCharging[month, period])
    model.DischargeControl = Constraint(model.T, rule=lambda m, month, period: m.BatteryDischarge[month, period] <= big_M * (1 - m.IsCharging[month, period]))
    
    def init_soc_rule(m, month, period): 
        if (month, period) == (1, 1): 
            return m.BatterySOC[month, period] == 0.5 * battery_capacity 
        return Constraint.Skip 
    model.InitialSOC = Constraint(model.T, rule=init_soc_rule)
    
    #SOC transition
    
    def soc_rule(m, month, period): 
        if (month, period) == (1, 1): 
            return Constraint.Skip 
        if period == 1: 
            prev = (month-1, 4) if month > 1 else (1, 1) 
        else: 
            prev = (month, period-1) 
        return m.BatterySOC[(month, period)] == m.BatterySOC[prev] + m.BatteryCharge[prev] - m.BatteryDischarge[prev] 
    model.SOC_Constraint = Constraint(model.T, rule=soc_rule)
    
    def soc_upper_limit(m, month, period):
        return m.BatterySOC[month, period] <= 0.9 * battery_capacity
    
    def soc_lower_limit(m, month, period):
        return m.BatterySOC[month, period] >= 0.2 * battery_capacity
    
    #model.SOCUpperLimit = Constraint(model.T, rule=soc_upper_limit)
    
    model.SOCLowerLimit = Constraint(model.T, rule=soc_lower_limit)
    
    #Energy balance
    
    def load_balance_rule(m,month, period):
        return (m.SolarGeneration[month, period] + m.EnergyBought[month, period] + m.BatteryDischarge[month, period] == m.Load[month, period] + m.EnergySold[month, period] + m.BatteryCharge[month, period])
    model.LoadBalance = Constraint(model.T, rule=load_balance_rule)
    
    #Charge only if SOC < 90% of capacity
    
    
    def charge_only_if_not_full(m, month, period):
        return m.BatteryCharge[month, period] <= ( battery_capacity - m.BatterySOC[month, period] ) 
    model.ChargeOnlyIfNotFull = Constraint(model.T, rule=charge_only_if_not_full)
    
    
    
    #Discharge only if SOC > 50% of capacity
    
    def discharge_only_if_not_empty(m, month,  period):
        return m.BatteryDischarge[month, period] <= m.BatterySOC[month,  period]+m.BatteryCharge[month, period]
    
    model.DischargeOnlyIfNotEmpty = Constraint(model.T, rule=discharge_only_if_not_empty)
    
    #Objective: minimize net cost and discourage excessive transfer
    
    
    def sell_limit(m, month,  period):
        return m.EnergySold[month, period] <= (m.SolarGeneration[month,  period] + m.BatteryDischarge[month,  period])
       
    
    
    model.SellLimit = Constraint(model.T, rule=sell_limit)
    
    
    def buy_limit(m, month,  period):
        return m.EnergyBought[month, period] <= (m.Load[month,  period])
        
    
    model.BuyLimit = Constraint(model.T, rule = buy_limit)
    
    #BDI
    
    
    # Set these based on your model or real-world specs
    expected_life_cycles = 6000 # typical for lithium-ion
    
    total_discharge = 0
    
    
    
    model.TotalCost = Objective( expr=sum( model.EnergyBought[month,  period] * model.BuyPrice[month,  period] - model.EnergySold[month,  period] * model.SellPrice[month,  period] + (model.EnergySold[month,  period] +  model.EnergyBought[month,  period])* 0.1 + battery_cycle_cost*(model.BatteryDischarge[month, period]/battery_capacity) for (month, period) in model.T ), sense=minimize )
    #Solving
    
    solver = SolverFactory('/home/majeed/miniconda3/envs/optenv/bin/ipopt') 
    solver.options['print_level'] = 3
    results2 = solver.solve(model, tee=False)
    #Output sample
    
    # ---------------------------
    # Output Results
    # ---------------------------
    

    from pyomo.environ import value
    
    data = []
    for (month, period) in sorted(model.T):
        buy_val = model.EnergyBought[month, period].value
        sell_val = model.EnergySold[month, period].value
        solar_val = model.SolarGeneration[month, period]
        load_val = model.Load[month, period]
        buy_price_val = model.BuyPrice[month, period].value
        sell_price_val = model.SellPrice[month, period].value
        charge_val = model.BatteryCharge[month, period].value
        discharge_val = model.BatteryDischarge[month, period].value
        soc_val = model.BatterySOC[month, period].value
        data.append({'Month': month, 'Period': period, 'Solar': solar_val, 'Load': load_val, 'Buy': buy_val, 'Sell': sell_val, 'Buy Price': buy_price_val, 'Sell Price': sell_price_val,  'Charge': charge_val, 'Discharge': discharge_val, 'SoC': soc_val, 'Total1': load_val + charge_val + sell_val,  'Total2': solar_val + discharge_val + buy_val})
        df = pd.DataFrame(data)
        df.to_latex("results_table.tex", index = False, longtable = True, float_format = "%.2f")
        
    
        # Create figure and axis

    
    
    # Extract interval data
    intervals = list(model.T.data()) # List of (month, period)
    
    # Prepare data for plotting
    x_labels = []
    gen_values = []
    sold_values = []
    bought_values = []
    
    for m, p in intervals:
        label = f"{m:02d}-{p}" # e.g., "01-1" for January, period 1
        x_labels.append(label)
        gen_values.append(model.SolarGeneration[m, p])
        sold_values.append(model.EnergySold[m, p].value)
        bought_values.append(model.EnergyBought[m, p].value)
    
    
    
    # Prepare data for plotting
    x_labels = []
    BatterySOC = []
    BatteryCharge = []
    BatteryDischarge = []
    
    
    for m, p in intervals:
        label = f"{m:02d}-{p}" # e.g., "01-1" for January, period 1
        x_labels.append(label)
        BatterySOC.append(model.BatterySOC[m, p].value)
        BatteryCharge.append(model.BatteryCharge[m, p].value)
        BatteryDischarge.append(model.BatteryDischarge[m, p].value)
    
    
    
  
       
    # Prepare data for plotting
    x_labels = []
    SellPrice = []
    BuyPrice = []
    
    for m, p in intervals:
        label = f"{m:02d}-{p}" # e.g., "01-1" for January, period 1
        x_labels.append(label)
        SellPrice.append(model.SellPrice[m, p].value)
        BuyPrice.append(model.BuyPrice[m, p].value)
    
       
    #ssei
    from pyomo.environ import value
    generated = 0
    solar_used = 0
    total_load = 0
    for (m, i) in model.T:
        gen = value(model.SolarGeneration[m, i])
        sold = value(model.EnergySold[m, i])
        charge = value(model.BatteryCharge[m, i])
        load = value(model.Load[m, i]) 
        # Solar used to meet direct load (not sold or stored)
        generated += gen
        used = max(0, gen - sold - charge)
        solar_used += used
        total_load += load
    ssi = solar_used / total_load if total_load > 0 else 0
    
   
    
    #GEDI
    
    
    grid_import = 0
    total_load = 0
    
    for (m, i) in model.T:
        bought = value(model.EnergyBought[m, i])
        load = value(model.Load[m, i])   
        grid_import += bought
        total_load += load
    gdi = grid_import / total_load if total_load > 0 else 0
    
    
    # BDI
    for (m, i) in model.T:
        discharge = value(model.BatteryDischarge[m, i])
        total_discharge += discharge
        cycle = total_discharge / battery_capacity
        cycleCost = cycle * cost_battery / 6000
    
    # Calculate equivalent full cycles used
    equivalent_cycles = total_discharge *30/ battery_capacity
    battery_lifetime = 6000/ equivalent_cycles
    
    bdi = equivalent_cycles / expected_life_cycles
   
    #LCSE
    
    # Capital costs (in USD)
    
    # Average annual solar energy generation (kWh)
    # We use actual model result summed over one year
    total_solar_generation_yearly = 30*sum(value(model.SolarGeneration[m, i]) for (m, i) in model.T)
    
    # Scale energy if model runs only for a part of the year
    # For full year, no scaling needed
    total_solar_lifetime = total_solar_generation_yearly * system_lifetime
    
    # Annualized battery and inverter replacement (based on shorter lifespan)
    battery_replacements = system_lifetime // battery_lifetime
    inverter_replacements = system_lifetime // inverter_lifetime
    
    # Total cost over system lifetime
    total_capital_cost = (
        cost_panels + 
        cost_container + 
        cost_stc_switch +
        cost_battery * battery_replacements +
        cost_inverter * inverter_replacements
    )
    # 
    annual_mc_cost = total_capital_cost * 0.1
    total_mc_cost = annual_mc_cost * system_lifetime
    total_cost = total_capital_cost + total_mc_cost 
    
    # Compute LCSE
    lcse = total_cost / total_solar_lifetime if total_solar_lifetime > 0 else float('inf')
    
    
    #levelized cost of Grid Energy
    
    
    # Initialize totals
    total_energy_bought = 0.1
    total_energy_sold = 0.1
    total_cost_bought = 0.1
    total_revenue_sold = 0.1
    total_cost_consumed = 0.1
    total_energy_consumed = 0.1
    
    # Sum over all intervals
    for (m, i) in model.T:
        eb = value(model.EnergyBought[m, i])
        es = value(model.EnergySold[m, i])
        ec = value(model.Load[m, i] - model.EnergyBought[m, i])
        bp = value(model.BuyPrice[m, i])
        sp = value(model.SellPrice[m, i])
        total_energy_bought += eb
        total_energy_sold += es
        total_energy_consumed += ec
        total_cost_bought += eb * bp
        total_cost_consumed += ec * lcse
        total_revenue_sold += es * (sp-lcse)
    
    # Compute LCGE 
    if total_energy_bought > 0:
        lcge = (total_cost_bought + total_cost_consumed - total_revenue_sold ) / (total_energy_bought + total_energy_consumed)
        
    else:
        lcge = float('inf') # or 0, depending on how you want to interpret it
    
   
   

    ELR = total_energy_sold / total_load
    EGR = total_energy_sold / generated
    BLR = total_discharge / total_load
    GLR = generated / total_load
    
    values = [ bdi, lcse, lcge, ssi, gdi, ELR , EGR , BLR, GLR]
    return lcge
import itertools
# Define ranges for sensitivity analysis
battery_capacities = [100, 200, 300, 400, 500]
Panels_num = [ 100, 200, 300, 400, 500]

# Store results
results = []
# Loop over all combinations
for battery_cap, panels in itertools.product(battery_capacities, Panels_num):
    output = run_microgrid_model( battery_cap,  panels)
    results.append(output)
# Convert to DataFrame
df = pd.DataFrame([results[i:i+5]for i in range(0,25,5)], index = battery_capacities, columns  = Panels_num)
# Save to Excel or CSV if needed
df.to_csv("sensitivity_results_bdi.csv", index=False)

#Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
 #Axis labels and title
plt.title("Levelised Cost of Electricity (Locally Consumed + Bought - Exported Electricity )")
plt.ylabel("Battery Capacity (kWh)")
plt.xlabel("Number of Solar Panels (0.72kw panel)")
plt.tight_layout()
plt.show()
