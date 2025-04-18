import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary

# ========================
# 1. Cargar y transformar datos
# ========================

# Leer archivo de Excel
xls = pd.ExcelFile('micron_db.xlsx')
df_raw = xls.parse('Supply_Demand')

# Limpiar y transformar
df_long = df_raw.iloc[2:].copy()
df_long.columns = df_raw.iloc[1]
df_long.rename(columns={df_long.columns[0]: 'Product ID', 'Attribute': 'Attribute'}, inplace=True)
df_melted = df_long.melt(id_vars=['Product ID', 'Attribute'], var_name='Period', value_name='Value')
df_pivot = df_melted.pivot_table(index=['Product ID', 'Period'], columns='Attribute', values='Value').reset_index()

# Convertir columnas a numéricas
numeric_columns = ['EffectiveDemand', 'Yielded Supply', 'Safety Stock Target', 'Safety Stock Target (WOS)']
for col in numeric_columns:
    df_pivot[col] = pd.to_numeric(df_pivot.get(col, 0), errors='coerce').fillna(0)

# ========================
# 2. Parámetros del modelo
# ========================

# Costos y capacidades
PRODUCTION_COST = 1.0    # Costo base de producción por unidad
HOLDING_COST = 0.1      # Costo de mantener inventario por unidad
SETUP_COST = 1000       # Costo de preparar la línea para un producto
MIN_PRODUCTION = 100000  # Producción mínima por lote
MAX_PRODUCTION = 300000  # Producción máxima por lote (reducida)
CAPACITY_PERIOD = 800000  # Capacidad total por período (reducida)
INITIAL_INVENTORY = 0    # Inventario inicial

# ========================
# 3. Construir modelo
# ========================

model = LpProblem("Production_Balance_Optimization", LpMinimize)

products = df_pivot['Product ID'].unique()
periods = df_pivot['Period'].unique()
periods_list = list(periods)

# ========================
# 4. Variables de decisión
# ========================

# Producción en cada período
production = {(i, t): LpVariable(f"prod_{i}_{t}", lowBound=0) for i in products for t in periods}
# Inventario al final de cada período
inventory = {(i, t): LpVariable(f"inv_{i}_{t}", lowBound=0) for i in products for t in periods}
# Variable binaria para setup
setup = {(i, t): LpVariable(f"setup_{i}_{t}", cat=LpBinary) for i in products for t in periods}
# Demanda cumplida
fulfilled = {(i, t): LpVariable(f"fulfilled_{i}_{t}", lowBound=0) for i in products for t in periods}

# ========================
# 5. Función objetivo
# ========================

# Minimizar costos totales: producción + inventario + setup
total_cost = (
    lpSum(PRODUCTION_COST * production[i, t] for i in products for t in periods) +  # Costo de producción
    lpSum(HOLDING_COST * inventory[i, t] for i in products for t in periods) +      # Costo de inventario
    lpSum(SETUP_COST * setup[i, t] for i in products for t in periods)              # Costo de setup
)

model += total_cost, "Total_Cost"

# ========================
# 6. Restricciones
# ========================

# Restricción de capacidad por período
for t in periods:
    model += lpSum(production[i, t] for i in products) <= CAPACITY_PERIOD, f"Capacity_{t}"

# Balance de inventario
for i in products:
    for t in periods:
        t_idx = periods_list.index(t)
        if t_idx == 0:
            # Inventario inicial
            prev_inv = INITIAL_INVENTORY
        else:
            prev_inv = inventory[i, periods_list[t_idx-1]]
        
        demand = float(df_pivot[(df_pivot['Product ID'] == i) & (df_pivot['Period'] == t)]['EffectiveDemand'].iloc[0])
        safety_stock = float(df_pivot[(df_pivot['Product ID'] == i) & (df_pivot['Period'] == t)]['Safety Stock Target'].iloc[0])
        
        # Balance de inventario
        model += inventory[i, t] == prev_inv + production[i, t] - fulfilled[i, t], f"Inv_Balance_{i}_{t}"
        # Cumplimiento de demanda
        model += fulfilled[i, t] <= demand, f"Demand_{i}_{t}"
        # Inventario mínimo (safety stock)
        model += inventory[i, t] >= safety_stock, f"Safety_Stock_{i}_{t}"

# Restricciones de producción mínima y máxima
for i in products:
    for t in periods:
        # Si hay producción, debe ser al menos MIN_PRODUCTION
        model += production[i, t] >= MIN_PRODUCTION * setup[i, t], f"Min_Prod_{i}_{t}"
        # Si hay producción, no puede exceder MAX_PRODUCTION
        model += production[i, t] <= MAX_PRODUCTION * setup[i, t], f"Max_Prod_{i}_{t}"
        # Si hay producción, setup debe ser 1
        model += production[i, t] <= MAX_PRODUCTION * setup[i, t], f"Setup_Link_1_{i}_{t}"
        # Si no hay producción, setup debe ser 0
        model += production[i, t] >= MIN_PRODUCTION * setup[i, t], f"Setup_Link_2_{i}_{t}"

# ========================
# 7. Resolver modelo
# ========================

model.solve()

# ========================
# 8. Resultados
# ========================

results = pd.DataFrame([
    (i, t, 
     production[i, t].varValue,
     inventory[i, t].varValue,
     fulfilled[i, t].varValue,
     setup[i, t].varValue)
    for (i, t) in production
], columns=["Product", "Period", "Produced", "Inventory", "Fulfilled", "Setup"])

print("\nResultados del modelo:")
print(results.head(10))
print("\nValor objetivo (costo total):", model.objective.value())

# Calcular estadísticas adicionales
print("\nEstadísticas por producto:")
for product in products:
    product_data = results[results['Product'] == product]
    print(f"\nProducto {product}:")
    print(f"Producción total: {product_data['Produced'].sum():,.0f}")
    print(f"Inventario promedio: {product_data['Inventory'].mean():,.0f}")
    print(f"Demanda cumplida: {product_data['Fulfilled'].sum():,.0f}")
    print(f"Períodos con producción: {product_data['Setup'].sum():.0f}")

results.to_csv("production_plan.csv", index=False)