import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# ========================
# 1. Cargar y transformar datos
# ========================

# Leer archivo de Excel
xls = pd.xlsx('micron_db.xlsx')
df_raw = xls.parse('Supply_Demand')

# Transformar la tabla: de ancha a larga (melt)
df_long = df_raw.iloc[2:].copy()
df_long.columns = df_raw.iloc[1]
df_long.rename(columns={df_long.columns[0]: 'Product ID', 'Attribute': 'Attribute'}, inplace=True)
df_melted = df_long.melt(id_vars=['Product ID', 'Attribute'], var_name='Period', value_name='Value')

# Pivotear para tener columnas por atributo
df_pivot = df_melted.pivot_table(index=['Product ID', 'Period'], columns='Attribute', values='Value').reset_index()

# ========================
# 2. Construir modelo de optimización
# ========================

model = LpProblem("Production_Balance_Optimization", LpMinimize)

# ========================
# 3. Variables de decisión
# ========================

products = df_pivot['Product ID'].unique()
periods = df_pivot['Period'].unique()

# Ejemplo: cantidad a producir por producto y periodo
x = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0) for i in products for t in periods}

# ========================
# 4. Función objetivo
# ========================

# Este ejemplo minimiza el exceso total de inventario sobre Safety Stock
excess_cost = 0
for _, row in df_pivot.iterrows():
    product, period = row['Product ID'], row['Period']
    sst = row.get('Safety Stock Target', 0)
    supply = x[(product, period)]
    excess = supply - sst
    excess_cost += excess

model += excess_cost

# ========================
# 5. Restricciones (ejemplos)
# ========================

# Capacidad máxima por período (puedes sacarla de otra hoja si aplica)
# model += lpSum(x[(i, t)] for i in products) <= capacidad[t]

# ========================
# 6. Resolver modelo
# ========================

model.solve()

# ========================
# 7. Resultados
# ========================

results = pd.DataFrame([(i, t, x[(i, t)].varValue) for (i, t) in x], columns=["Product", "Period", "Produced"])
print(results.head())