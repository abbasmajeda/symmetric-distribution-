#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:04:41 2025

@author: majeed
"""
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
discount_rate = 0.08
dt=6
T_avg = 30        # average battery temp

# C-rate
C_ref = 0.5       # nominal (0.5C = 2h charge)
C_avg = 1.0       # average operating rate

# Sensitivity coefficients (typical)
k_T = 0.07        # temp aging factor
k_C = 0.3         # C-rate aging factor
pv_deg = 0.005
import math

f_T = math.exp(k_T * (T_avg - Tref))
f_C = 1 + k_C * (C_avg - C_ref)
aging_factor = 1 / (f_T * f_C)

#TOU Buy price Design
tou_pattern = {
    1: "off",
    2: "mid",
    3: "peak",
    4: "peak"
}
tou_price = {
    "off": 0.05,
    "mid": 0.076,
    "peak": 0.1
}

time_factor = {
    1: 0.7,
    2: 1.0,
    3: 1.3,
    4: 1.5
}
def season_factor(month):
    if month in [6,7,8]:
    	return 1.2
    elif month in [12,1,2]:
    	return 1.3	
    else:
    	return 1.0	

def discount_factor(r, y):
    return 1 / ((1 + r) ** y)

 
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
    Battery_Cycles = 8000
    DoD = 0.8        # usable depth of discharge
    C_rate = 0.12
    P_max_per_period = C_rate * battery_capacity * 6  # kWh per 6h period


    # Expected component lifetimes (in years)
    panel_lifetime = 25
    inverter_lifetime = 5
    system_lifetime = 25 # analysis period

    #load profile
    df_load = pd.read_csv("load_profile_long.csv")
    load = df_load.set_index(['Month','Period_num'])['Load'].to_dict()

    # Grid connection limit (kW per period)
    Grid_import_max = 2 * max(load.values())
    Grid_export_max = 2 * max(load.values())


    # Inverter rating (kW)
    PV_Max = solar_panel_count * wh_per_hour * dt


    for month in range(1, 13): 
        #total_gen = sunshine_hours[month - 1] * solar_panel_count * wh_per_hour *( 1 + Temp_coef * (T_ref -temperature[month - 1])) 
        solar_generation[(month, 1)] = 0.5 * compute_monthly(month) * solar_panel_count * wh_per_hour
        solar_generation[(month, 2)] = 0.5 * compute_monthly(month) * solar_panel_count * wh_per_hour
        solar_generation[(month, 3)] = 0 # No solar generation at night
        solar_generation[(month, 4)] = 0
  
    model.SolarAvailable = Param(model.T, initialize=solar_generation)
    model.SolarUsed = Var(model.T, within=NonNegativeReals)
    model.Load = Param(model.T, initialize=load)
    
    #Battery parameters   

    model.ChargeAllowed = Var(model.T, domain=Binary)
    M = battery_capacity # Big M value, or max possible charge
    R_line = 0.00002
    
    #Toubuy price
    BuyPrice = {}
    for m, p in model.T:
       	block = ((p-1) % 4) + 1
       	level = tou_pattern[block]
       	BuyPrice[(m,p)] = tou_price[level]

    #sell price dynamic market base with seasonl, time factor and market uncertainity
    import random
    random.seed(1)
    base_sell_price=0.045
    SellPrice = {}

    for m, p in model.T:
    	season = season_factor(m)
    	block = ((p-1) % 4) + 1
    	tmfctr= time_factor[block]
    	noise = random.uniform(0.9, 1.1)
    	price = base_sell_price * tmfctr * season * noise
    	# Limit selling profit
    	price = min(price, 1.1 * BuyPrice[(m,p)])
    	price = max(price, 0.01)
    	SellPrice[(m, p)] = price

    #Variables
    model.EnergyBought = Var(model.T, within=NonNegativeReals) 
    model.EnergySold = Var(model.T, within=NonNegativeReals) 
    model.BatterySOC = Var(model.T, bounds=(0.2* battery_capacity, 0.95* battery_capacity)) 
    model.BatteryCharge = Var(model.T, within=NonNegativeReals) 
    model.BatteryDischarge = Var(model.T, within=NonNegativeReals)
    model.IsCharging = Var(model.T, domain=Binary)
    model.IsImport = Var(model.T, domain=Binary)
    model.SolarCurtail = Var(model.T, domain=NonNegativeReals)
    model.BatteryTemp = Var(model.T, bounds=(15, 45))

    # Voltage at the bus (per period)
    model.Vbus = Var(model.T, bounds=(0.95, 1.05))  # bounds in p.u.

    model.eta_c = Param(initialize=0.95)   # charging efficiency
    model.eta_d = Param(initialize=0.95)   # discharging efficiency
    model.SOC_init_frac = Param(initialize=0.5)
    model.SOC0 = Param(initialize = 0.5 * battery_capacity)
    model.Dump = Var(model.T, domain=NonNegativeReals)

    eta_rt = model.eta_c * model.eta_d
    Lifetime_Energy = ( battery_capacity * DoD * Battery_Cycles * eta_rt * aging_factor)
    battery_cycle_cost = cost_battery / Lifetime_Energy
    

        
    #Dynamic prices (to be optimized)
    model.BuyPrice = Param(model.T, initialize=0.076) 
    model.SellPrice = Param(model.T, initialize= 0) 
        
    
    
    #Initial SOC 
    model.ChargeControl = Constraint(model.T, rule=lambda m, month, period: m.BatteryCharge[month, period] <= M * m.ChargeAllowed[month, period])
    model.DischargeControl = Constraint(model.T, rule=lambda m, month, period: m.BatteryDischarge[month, period] <= M * (1 - m.ChargeAllowed[month, period]))
    
    #solar used
    def solar_limit(m, month, period):
        return m.SolarUsed[month, period] <= m.SolarAvailable[month, period]
    model.SolarLimit = Constraint(model.T, rule=solar_limit)

    def solar_balance(m, month, period):
        return (m.SolarUsed[month, period] + m.SolarCurtail[month, period] == m.SolarAvailable[month, period]  )
    model.SolarBalance = Constraint(model.T, rule=solar_balance)
    
    #inverter constraint
    def pv_limit(m, month, period):
        return m.SolarUsed[month, period] <= PV_Max
    model.PVLimit = Constraint(model.T, rule=pv_limit)
    
    #grid constrain
    def grid_buy_control(m, month, period):
        return m.EnergyBought[month, period] <= Grid_import_max * m.IsImport[month,period]
    model.GridBuyLimit = Constraint(model.T, rule=grid_buy_control)

    def grid_sell_control(m, month, period):
        return m.EnergySold[month, period] <= Grid_export_max * (1 - m.IsImport[month,period])
    model.GridSellLimit = Constraint(model.T, rule=grid_sell_control)

    #SOC transition    
    def soc_rule(m, month, period): 
        if (month, period) == (1, 1):
            prev_soc = m.SOC0
        else:
            if period == 1: 
                prev_soc = m.BatterySOC[month-1, 4]
            else: 
                prev_soc = m.BatterySOC[month, period-1]
        return m.BatterySOC[(month, period)] == prev_soc + m.eta_c* m.BatteryCharge[month, period] - (1/m.eta_d)*m.BatteryDischarge[month, period] 
    model.SOC_Constraint = Constraint(model.T, rule=soc_rule)
    
   
   
    #Energy balance
    
    def load_balance_rule(m,month, period):
        return (m.SolarUsed[month, period] + m.EnergyBought[month, period] + m.BatteryDischarge[month, period] == m.Load[month, period] + m.EnergySold[month, period] + m.BatteryCharge[month, period] + m.SolarCurtail[month, period] + m.Dump[month, period])
    model.LoadBalance = Constraint(model.T, rule=load_balance_rule)
    
    #Charge only if SOC < 90% of capacity
    
    
    def charge_limit(m, month, period):
        return m.BatteryCharge[month, period] <= P_max_per_period
    model.ChargeLimit = Constraint(model.T, rule=charge_limit)

    def discharge_limit(m, month, period):
        return m.BatteryDischarge[month, period] <= P_max_per_period
    model.DischargeLimit = Constraint(model.T, rule=discharge_limit)

    
    
    def buy_limit(m, month,  period):
        return m.EnergyBought[month, period] <= (m.Load[month,  period])
    model.BuyLimit = Constraint(model.T, rule = buy_limit)
    
    def inverter_limit(m,month,period):
        return (m.SolarUsed[month,period] + m.EnergySold[month,period]  <= PV_Max )
    model.InverterLimit = Constraint(model.T, rule=inverter_limit)

    Ramp = 0.5 * battery_capacity

    def charge_ramp(m, month, period):
        if period == 1 and month == 1:
            return Constraint.Skip
        prev = (month,period-1) if period >1 else (month-1,4)
        return abs(m.BatteryCharge[month,period] - m.BatteryCharge[prev]) <= Ramp
    model.RampLimit = Constraint(model.T, rule=charge_ramp)

    def voltage_constraint(m, month, period):
        return m.Vbus[month, period] == 1.0 - R_line * (m.Load[month, period] - m.SolarUsed[month, period] - (m.BatteryDischarge[month, period] - m.BatteryCharge[month, period])) / 100  # scale if needed
    model.VoltageConstraint = Constraint(model.T, rule=voltage_constraint)

    f_nom = 50  # Hz
    delta_f_max = 0.2  # Hz allowed deviation
    Kf = 0.001  # Hz per kW, adjust based on system size
    def frequency_response(m, month, period):
        # net power causing frequency deviation
        P_net = m.Load[month, period] - m.SolarUsed[month, period] - (m.BatteryDischarge[month, period] - m.BatteryCharge[month, period])   - m.EnergyBought[month, period] + m.EnergySold[month, period]
        return (-delta_f_max, Kf * P_net, delta_f_max)
    model.FrequencyConstraint = Constraint(model.T, rule=frequency_response)

      
    #expected_life_cycles = 6000 # typical for lithium-ion
    total_discharge = 0
    

    alpha = 0.02    # heating per kWh
    beta = 0.1      # cooling factor

    def temp_rule(m, month, period):
        if (month, period) == (1,1):
            return m.BatteryTemp[month,period] == Tref
        if period == 1:
            prev = (month-1,4) if month>1 else (1,1)
        else:
            prev = (month,period-1)
        heat = dt*alpha*(m.BatteryCharge[month,period] + m.BatteryDischarge[month,period])
        cool = dt*beta*(m.BatteryTemp[prev] - Tref)
        return m.BatteryTemp[month,period] == m.BatteryTemp[prev] + heat - cool
    model.TempLimit = Constraint(model.T, rule=lambda m,mo,p: m.BatteryTemp[mo,p] <= 40)

    model.TemperatureConstraint = Constraint(model.T, rule=temp_rule)
    
    model.TotalCost = Objective( expr=sum( model.EnergyBought[month,  period] * model.BuyPrice[month,  period] - model.EnergySold[month,  period] * model.SellPrice[month,  period] + (model.EnergySold[month,  period] +  model.EnergyBought[month,  period])* 0.1 + battery_cycle_cost * model.BatteryDischarge[month, period] + 0.1 * model.SolarCurtail[month,  period] + 1e4 * sum(model.Dump[t] for t in model.T)
 for (month, period) in model.T ), sense=minimize )
    #Solving
    
    solver = SolverFactory('/home/majeed/miniconda3/envs/optenv/bin/bonmin') 
    results = solver.solve(model, tee=True)
    #Output sample
    max_freq_dev = 0
    from pyomo.environ import value

    print("SOC min:", min(value(model.BatterySOC[t]) for t in model.T))
    print("SOC max:", max(value(model.BatterySOC[t]) for t in model.T))

    print("Charge sum:", sum(value(model.BatteryCharge[t]) for t in model.T))
    print("Discharge sum:", sum(value(model.BatteryDischarge[t]) for t in model.T))

    print("Curtailment:", sum(value(model.SolarCurtail[t]) for t in model.T))
    print("Export:", sum(value(model.EnergySold[t]) for t in model.T))



    for (month, period) in model.T:
        P_net = ( model.Load[month, period] - model.SolarUsed[month, period].value - (model.BatteryDischarge[month, period].value  - model.BatteryCharge[month, period].value)  - model.EnergyBought[month, period].value   + model.EnergySold[month, period].value )
        freq_dev = abs(Kf * P_net)
        if freq_dev > max_freq_dev:
            max_freq_dev = freq_dev
        print(month,period, model.BatteryTemp[month,period].value)

   
    # ---------------------------
    # Output Results
    # ---------------------------
   
    from pyomo.environ import value
    
    total_energy_bought = 0
    total_energy_sold = 0
    total_cost_bought = 0
    total_revenue = 0
    total_cost_consumed = 0
    annual_grid_cost = 0
    annual_revenue = 0
    annual_load = 0
    lifetime_revenew =0
    lifetime_load = 0
    total_load = 0
    generated = 0
    data = []
    for (month, period) in sorted(model.T):
        buy_val = model.EnergyBought[month, period].value
        sell_val = model.EnergySold[month, period].value
        solar_val = model.SolarUsed[month, period].value
        solar_gen = model.SolarAvailable[month, period]
        load_val = model.Load[month, period]
        buy_price_val = model.BuyPrice[month, period]
        sell_price_val = model.SellPrice[month, period]
        charge_val = model.BatteryCharge[month, period].value
        discharge_val = model.BatteryDischarge[month, period].value
        soc_val = model.BatterySOC[month, period].value
        generated += solar_gen
        data.append({'Month': month, 'Period': period, 'Solar': solar_val, 'Solar_gen': solar_gen, 'Load': load_val, 'Buy': buy_val, 'Sell': sell_val, 'Buy Price': buy_price_val, 'Sell Price': sell_price_val,  'Charge': charge_val, 'Discharge': discharge_val, 'SoC': soc_val, 'Total1': load_val + charge_val + sell_val,  'Total2': solar_val + discharge_val + buy_val})
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
        gen_values.append(model.SolarUsed[m, p].value)
        sold_values.append(model.EnergySold[m, p].value)
        bought_values.append(model.EnergyBought[m, p].value)
    
    # Create x-axis positions
    x = np.arange(len(x_labels))
    width = 0.25
    
    # Plot
    plt.figure(figsize=(18, 6))
    plt.bar(x - width, gen_values, width, label='Generated', color='gold')
    plt.bar(x, sold_values, width, label='Sold', color='green')
    plt.bar(x + width, bought_values, width, label='Bought', color='red')
    
    # Labeling and layout
    plt.xticks(x, x_labels, rotation=90)
    plt.xlabel("Month-Period")
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Generated, Sold, and Bought per Interval (Month-Period)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # Create figure 
    plt.figure(figsize=(10, 6))
    
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
    
    
    # Plot SellPrice on left y-axis
    plt.xlabel("Month - Period")
    plt.ylabel( "Energy (kWh)", color='orange')
        
    plt.bar(x - width, BatterySOC, width, label='Battery SOC', color='gold')
    plt.bar(x, BatteryCharge, width, label='Battery Charge', color='green')
    plt.bar(x + width, BatteryDischarge, width, label='Battery Discharge', color='red')
    
    # Title and layout
    plt.title("Battery System Situation")
    plt.xticks(x, x_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Show plot
    plt.show()



    # Create figure 
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    x_labels = []
    SellPrice = []
    BuyPrice = []
    
    for m, p in intervals:
        label = f"{m:02d}-{p}" # e.g., "01-1" for January, period 1
        x_labels.append(label)
        SellPrice.append(model.SellPrice[m, p])
        BuyPrice.append(model.BuyPrice[m, p])
    
    # Plot SellPrice on left y-axis
    plt.xlabel("Month - Period")
    plt.ylabel(" Price (per KWh)", color='orange')
    plt.plot( x_labels, SellPrice, label ="Sell Price", marker='o', color='orange')
    
    # Plot BuyPrice on right y-axis
    plt.plot( x_labels, BuyPrice, label ="Buy Price", marker='s', color='crimson')
    
    # Title and layout
    plt.title("Buying and Selling Prices")
    plt.xticks(x, x_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Show plot
    plt.show()
    
    # BDI
    calendar_loss = 0.015
    total_discharge = 0.0
    for (m, i) in model.T:
        discharge = value(model.BatteryDischarge[m, i])
        T     = value(model.BatteryTemp[m,p])
        fT = math.exp(k_T*(T - Tref))
        total_discharge += discharge *fT
    bdi = total_discharge *30/ (battery_capacity *  Battery_Cycles* (0.8 / DoD)**0.5)
    battery_lifetime = 1/bdi
    print(f"Battery Depreciation Index (BDI): {bdi:.4f}")
    
    #LCSE
    
    # Capital costs (in USD)
    
    # Average annual solar energy generation (kWh)
    # We use actual model result summed over one year
    total_solar_generation_yearly = 30*sum(
    						min(
    							model.SolarUsed[m, i].value, model.Load[m, i]
    						)
    						+ model.BatteryDischarge[m,i].value * model.eta_d
    					 for (m, i) in model.T)
    

    # For full year
    total_solar_lifetime = sum( total_solar_generation_yearly * (1 - pv_deg)**(y-1) * discount_factor(discount_rate,y) for y in range(1,system_lifetime+1))

    
    # Annualized battery and inverter replacement (based on shorter lifespan)
    battery_replacements = system_lifetime / battery_lifetime
    inverter_replacements = system_lifetime / inverter_lifetime
    
    # Total cost over system lifetime
    total_capital_cost = (
        cost_panels + 
        cost_container + 
        cost_stc_switch +
        cost_battery * battery_replacements +
        cost_inverter * inverter_replacements
    )
    # 
    annual_mc_cost = total_capital_cost * 0.01
    total_mc_cost = 0
    for y in range(1, system_lifetime + 1):
    	total_mc_cost += annual_mc_cost * discount_factor(discount_rate, y)
    total_cost = total_capital_cost + total_mc_cost 
    
    # Compute LCSE
    lcse = total_cost / total_solar_lifetime if total_solar_lifetime > 0 else float('inf')
    
    print(f"Levelized Cost of Solar Energy (LCSE): ${lcse:.4f} per kWh")
    
    #levelized cost of Grid Energy
    
    
    # Initialize totals 
    annual_bought = 0
    lifetime_bought = 0
    lifetime_grid_cost = 0
    # Sum over all intervals
    for (m, i) in model.T:
        eb = value(model.EnergyBought[m, i])
        es = value(model.EnergySold[m, i])
        l = value(model.Load[m, i] )
        bp = value(model.BuyPrice[m, i])
        sp = value(model.SellPrice[m, i])
        total_energy_bought += eb
        total_energy_sold += es
        total_cost_bought += eb * bp
        total_load += l
        total_cost_saved = total_load *bp
        total_revenue += es * (sp)
    annual_grid_cost = total_cost_bought *30
    annual_revenue = total_revenue *30
    annual_load = total_load *30
    annual_bought = total_energy_bought * 30
    for y in range(1, system_lifetime + 1):
        lifetime_revenew += annual_revenue * discount_factor(discount_rate,y)
        lifetime_load += annual_load * discount_factor(discount_rate,y)
        lifetime_grid_cost += annual_grid_cost * discount_factor(discount_rate,y)
    # Compute NECI 
    NECI = (total_cost - lifetime_revenew + lifetime_grid_cost) / (lifetime_load)
        
    
   #ssei 
    ssi = total_solar_generation_yearly / annual_load   
    print(f"Solar Self-Sufficiency Index (SSI): {ssi:.2%}")
   
    #GEDI
    gdi = total_energy_bought / total_load 
    print(f"Grid Dependency Index (GDI): {gdi:.2%}")
   
    #calculate NPV    
    discounted_cash_flows = []
    for y in range(1, 26):
    	discounted_cash_flows.append((total_solar_generation_yearly*bp + annual_revenue) * discount_factor(discount_rate,y))

    npv = sum(discounted_cash_flows) - total_cost
    
    #calculate payback period
    cumulative_disc = 0
    discounted_payback = 0
    for i, dcf in enumerate(discounted_cash_flows, start=1):
    	cumulative_disc += dcf
    	if cumulative_disc >= total_cost:
    	    prev_disc = cumulative_disc - dcf
    	    fraction = (total_cost - prev_disc)/dcf
    	    discounted_payback = i - 1 + fraction
    	    break
    
    # Output
    print(f" Net Energy Cost Indes (NECI): ${NECI:.4f} per kWh")
    print(f"Total Energy Bought: {total_energy_bought:.2f} kWh")
    print(f"Total Energy Sold: {total_energy_sold:.2f} kWh")
    print(f"Total Cost of Energy Bought: ${total_cost_bought:.2f}")
    print(f"Total Revenue from Energy Sold: ${total_revenue:.2f}")
    
    #indexes plot
    
    # ---- Labels and Values ----
    index_labels = [
        "Battery Depreciation",
        "LCSE ($/kWh)",
        "NECI ($/kWh)",
        "Net Present Value (100000$)",
        "Payback Period (10 Year)"
    ]
    
    index_values = [
        bdi,
        lcse,
        NECI,
        npv/100000,
        discounted_payback/10
    ]
    
    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(index_labels, index_values, color=['#FFB347', '#90EE90', '#ADD8E6', '#FFD700', '#FFA07A'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), # offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # ---- Formatting ----
    ax.set_title("Micro-grid Performance Indexes")
    ax.set_ylabel("Index Value")
    ax.set_ylim(0, max(index_values) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation = 15, fontsize=8)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.show()
    
    labels = [
        "Export-to-Load Ratio (ELR)",
        "Export-to-Generation Ratio (EGR)",
        "Battery-to-Load Ratio (BLR)", 
        "Generation-to-Load Ratio (BLR)",
        "Solar Sufficiency Index (SSI)"
    ]
    ELR = total_energy_sold / total_load
    if(generated > 0): EGR = total_energy_sold / generated
    BLR = total_discharge / total_load
    GLR = generated / total_load
    
    values = [ ELR , EGR , BLR, GLR, ssi]
    
    # --- Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#FFB347","#90EE90", "#ADD8E6", "#FFD700", "#FFA07A"])
    
    # Annotate each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2%}",
                     xy=(bar.get_x() + bar.get_width() / 2, height/2),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.ylabel("Ratio")
    plt.title("Microgrid Energy Utilization Ratios")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    
    # --- Step 1: Dictionary of performance indexes (use your computed values here)
    index_data = {
        "Battery Depreciation Index (BDI)": round(bdi, 3),
        "Levelized Cost of Solar Energy (LCSE) \$\/kWh": round(lcse, 3),
        "Net Energy Cost Index (NECI) \$\/kWh": round(NECI, 3),
        "Energy Independence Index (EII)": f"{ssi:.2}",
        "Grid Dependency Index (GDI)": f"{gdi:.2}",
        "Export-to-Load Ratio (ELR)": f"{ELR:.2}",
        "Export-to-Generation Ratio (EGR)": f"{EGR:.2}",
        "Battery-to-Load Ratio (BLR)": f"{BLR:.2}",
        "Generation-to-Load Ratio (GLR)": round(GLR, 3),
        "Net Present Value (NPV)": round(npv, 3),
        "Payback Period": round(discounted_payback, 3),
    }
    
    # --- Step 2: Convert to DataFrame
    df_summary = pd.DataFrame(index_data.items(), columns=["Index Name", "Value"])
    
    # --- Step 3: Export to LaTeX
    df_summary.to_latex(
        "microgrid index summary SO.tex",
        index=False,
        caption="Microgrid Performance Index Summary",
        label="tab:microgrid summary",
        longtable=False,
        escape=False # Allows math formatting like \$ or \%
    )
     # Create figure 
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    x_labels = []
       
    for m, p in intervals:
        label = f"{m:02d}-{p}" # e.g., "01-1" for January, period 1
        x_labels.append(label)
        
    # Plot SellPrice on left y-axis
    plt.xlabel("Month - Period")
    plt.ylabel(" Load (per KWh)", color='orange')
    plt.plot( x_labels, load.values(), label ="load", marker='o')
   
    # Title and layout
    plt.title("Period Load Profile for One Day in Each Month for 100 Houses")
    plt.xticks(x, x_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Show plot
    plt.show()
    
def plot_sun_shine_temperature():        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot sunshine hours on left y-axis
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sunshine Hours", color='orange')
    ax1.plot(months, sunshine_hours, marker='o', color='orange', label="Sunshine Hours")
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.set_ylim(0, max(sunshine_hours) + 2)
    
    # Plot temperature on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature (°C)", color='crimson')
    ax2.plot(months, temperature, marker='s', color='crimson', label="Temperature")
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, max(temperature) + 5)
    
    # Title and layout
    plt.title("Monthly Sunshine Hours and Temperature")
    fig.tight_layout()
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    
    # Show plot
    plt.show()

run_microgrid_model(300, 300)
#plot_sun_shine_temperature()

