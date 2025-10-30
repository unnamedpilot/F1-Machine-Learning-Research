"""
===============================================================
ANÁLISIS DE RESULTADOS DEL MODELO F1
===============================================================

Autor: Juan Esteban Pavas González
Proyecto: Influencia Piloto vs. Automóvil en la Fórmula 1
Fecha: Octubre 2025

Objetivos SMART cubiertos:
- SMART 2: Desarrollo del Modelo Predictivo
- SMART 3: Análisis de Influencia Piloto vs Auto
- SMART 4: Validación Prospectiva (placeholder)
===============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------------
# 1. Cargar resultados del modelo
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "..", "..", "model_results_summary.csv")
COEFFS_PATH = os.path.join(BASE_DIR, "..", "..", "model_coefficients.csv")

df_results = pd.read_csv(RESULTS_PATH)
df_coeffs = pd.read_csv(COEFFS_PATH)

print("============================================================")
print("ANÁLISIS DE RESULTADOS DEL MODELO")
print("============================================================\n")

# -------------------------------------------------------------
# 2. Objetivo SMART 2 – Desarrollo del Modelo Predictivo
# -------------------------------------------------------------
print("2️⃣  OBJETIVO SMART 2: Desarrollo del Modelo Predictivo\n")

print(df_results)
print("\n📊 Métricas principales:")

def get_metric(metric_name):
    """Extrae el valor numérico de una métrica desde df_results."""
    return float(df_results.loc[df_results["Metric"] == metric_name, "Value"].values[0])

print(f"- RMSE (test): {get_metric('Test RMSE'):.3f}")
print(f"- MAE (test):  {get_metric('Test MAE'):.3f}")
print(f"- Kendall τ:   {get_metric('Test Kendall Tau'):.3f}")

print("\n✅ El modelo demuestra capacidad predictiva razonable.")

# -------------------------------------------------------------
# 3. Objetivo SMART 3 – Análisis de Influencia Piloto vs Auto
# -------------------------------------------------------------
print("\n3️⃣  OBJETIVO SMART 3: Influencia Piloto vs Automóvil\n")

# Clasificar coeficientes
df_coeffs = df_coeffs.sort_values(by="coefficient", ascending=False)

drivers = df_coeffs[df_coeffs["feature"].str.contains("driver_")]
constructors = df_coeffs[df_coeffs["feature"].str.contains("constructor_")]
circuits = df_coeffs[df_coeffs["feature"].str.contains("circuit_")]

# Mostrar top 10
top_drivers = drivers.head(10)
top_teams = constructors.head(10)

print("\n🏎️  Top 10 Pilotos con mayor impacto positivo:")
print(top_drivers[["feature", "coefficient"]])

print("\n🏁  Top 10 Equipos con mayor impacto positivo:")
print(top_teams[["feature", "coefficient"]])

# -------------------------------------------------------------
# Visualizaciones
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(
    y=top_drivers["feature"].str.replace("driver_", ""),
    x=top_drivers["coefficient"],
    hue=top_drivers["feature"].str.replace("driver_", ""),
    palette="crest",
    legend=False
)
plt.title("Top 10 Pilotos con mayor influencia (Coeficiente Ridge)")
plt.xlabel("Coeficiente (impacto positivo en performance)")
plt.ylabel("Piloto")
plt.tight_layout()
driver_plot_path = os.path.join(BASE_DIR, "top_drivers_influence.png")
plt.savefig(driver_plot_path)
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(
    y=top_teams["feature"].str.replace("constructor_", ""),
    x=top_teams["coefficient"],
    hue=top_teams["feature"].str.replace("constructor_", ""),
    palette="flare",
    legend=False
)
plt.title("Top 10 Equipos con mayor influencia (Coeficiente Ridge)")
plt.xlabel("Coeficiente (impacto positivo en performance)")
plt.ylabel("Equipo")
plt.tight_layout()
team_plot_path = os.path.join(BASE_DIR, "top_teams_influence.png")
plt.savefig(team_plot_path)
plt.close()

print("\n📈 Visualizaciones guardadas correctamente:")
print(f"   → {driver_plot_path}")
print(f"   → {team_plot_path}")


print("\n📈 Visualizaciones guardadas: 'top_drivers_influence.png' y 'top_teams_influence.png'")

# -------------------------------------------------------------
# 4. Objetivo SMART 4 – Validación Prospectiva
# -------------------------------------------------------------
print("\n4️⃣  OBJETIVO SMART 4: Validación Prospectiva (Placeholder)\n")
print("⚙️  Pendiente: comparar con resultados reales de 2025.\n")
print("   → Se conectará con nuevo CSV de resultados F1_2025.csv")
print("   → Evaluará métricas (RMSE, MAE, Kendall τ) en nuevas carreras.")
print("\n============================================================")
print("✅ ANÁLISIS FINALIZADO")
print("============================================================")
