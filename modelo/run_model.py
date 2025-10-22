import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge #Esta libreria es para el modelo de regresión ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler #Esta libreria estandariza datos, ejemplo: [1, 5, 20]  →  [-1.2, 0.0, 1.2] o [2014, 2018, 2024]  →  [-1.0, 0.0, 1.0]
from sklearn.metrics import mean_squared_error, mean_absolute_error #Calcula errores, ejemplo: Posición real: [1, 2, 3] Posición predicha: [1, 3, 2] MAE = (0 + 1 + 1) / 3 = 0.67 posiciones de error promedio
from scipy.stats import kendalltau #Calcula Kendall's T
import warnings




warnings.filterwarnings('ignore')

print("="*70)
print("Influencia relativa del piloto y del automovil en el rendimiento en la Formula 1")
print("="*70)



# 1. Cargar datos
print("\n1. Cargando datos...")

df = pd.read_csv('F1_cleaned_for_model.csv')
print(f"Dataset cargado: {df.shape[0]:,} registros") #Muestra el numero de registros en el dataset



# 2. Preprocesar datos - Aca la idea es terminar de limpiar los datos
print("\n2. Preprocesamiento...")

df_hybrid = df[(df['year'] >= 2014) & (df['year'] <= 2024)].copy() #Filtrar los datos del 2014 al 2024
df_hybrid = df_hybrid[df_hybrid['positionOrder'].notna()] #Quita los datos que poisicion esta vacio o null
df_hybrid = df_hybrid[df_hybrid['positionText'] != 'R'] #Elimina filas donde tiene "R" que siginifica que se retiro (abandono)
df_hybrid = df_hybrid[df_hybrid['positionOrder'] <= 20] #Carreras que la posicion es 1-20

print(f"Era Híbrida (2014-2024): {len(df_hybrid):,} observaciones válidas")



# 3. FEATURE ENGINEERING
print("\n3. Feature Engineering...")


features_to_keep = [
    'driverId', 'constructorId', 'grid', 'positionOrder',
    'year', 'circuitId', 'raceId', 'laps'
] # Columnas que estaremos usando

df_model = df_hybrid[features_to_keep].copy() #Nueva tabla con las columnas nuevas
df_model['grid'].fillna(20, inplace=True) #Si grid esta Vacio se pone '20' ya que esto significa que salio desde pit


# Variables dummy
driver_dummies = pd.get_dummies(df_model['driverId'], prefix='driver', dtype=int) #Convierte el id en binario
constructor_dummies = pd.get_dummies(df_model['constructorId'], prefix='constructor', dtype=int) #Convierte el id del constructir (carro-equipo) en binario
circuit_dummies = pd.get_dummies(df_model['circuitId'], prefix='circuit', dtype=int)

print(f"Pilotos: {driver_dummies.shape[1]} categorías")
print(f"Automoviles: {constructor_dummies.shape[1]} categorías")
print(f"Circuitos: {circuit_dummies.shape[1]} categorías")


X_features = pd.concat([
    df_model[['grid', 'year', 'laps']],
    driver_dummies,
    constructor_dummies,
    circuit_dummies
], axis=1)


y_target = df_model['positionOrder']
print(f"Matriz de features: {X_features.shape}")




# 4. Split
print("\n4. Split Temporal...")


#Hace una lista True/Flase para saber si una fila sera de entrenamiento o de test, algo asi:   
''' year  train_mask  test_mask
0  2020        True      False
1  2021        True      False
'''


train_mask = df_model['year'] <= 2022 #Hasta 2021 aprende
test_mask = df_model['year'] >= 2023 #Desde 2022 evalua



X_train = X_features[train_mask]
X_test = X_features[test_mask]
y_train = y_target[train_mask]
y_test = y_target[test_mask]
train_race_ids = df_model.loc[train_mask, 'raceId']
test_race_ids = df_model.loc[test_mask, 'raceId']

print(f"Train (2014-2022): {X_train.shape[0]:,} observaciones")
print(f"Test (2023-2024): {X_test.shape[0]:,} observaciones")




# 5. Estandarizar
print("\n5. Estandarizar")


scaler = StandardScaler() #Esto le pone media=0 y desviación=1 para que todos tengan la misma "importancia"
numeric_cols = ['grid', 'year', 'laps']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols]) #Aprende de train y transforma
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols]) #Aplica la misma transformacion a test

print("   Features numéricas estandarizadas")




# 6. Entrenamiento
print("\n6. Entrenamiento del Modelo Ridge...")


# Alpha es el parámetro de regularización de Ridge. 
# Alpha bajo (0.001): Modelo flexible, riesgo de overfitting "muy flexible (acepta todo)"
# Alpha equilibrado (10)
# Alpha alto (1000): Modelo rígido, riesgo de underfitting "muy estricto (rechaza todo)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} #


tscv = TimeSeriesSplit(n_splits=5)
ridge = Ridge(random_state=42, max_iter=10000)

grid_search = GridSearchCV(
    ridge, param_grid, cv=tscv,
    scoring='neg_mean_squared_error', n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train) #  Busca en la pruebas alpha cual tiene el menor error y selecciona ese
best_alpha = grid_search.best_params_['alpha']
print(f"Mejor alpha: {best_alpha}")

best_ridge = Ridge(alpha=best_alpha, random_state=42, max_iter=10000)
best_ridge.fit(X_train_scaled, y_train) #Entrena el modelo con el mejor alpha encontrado
print("✅ Modelo entrenado")




# 7. Evaluacion
print("\n7. Evaluación del Modelo...")



#Usa el modelo entrenado para predecir posiciones.
y_train_pred = best_ridge.predict(X_train_scaled)
y_test_pred = best_ridge.predict(X_test_scaled)


#Calcula errores
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTEST SET (2023-2024):")
print(f"- RMSE: {np.sqrt(test_mse):.4f}")
print(f"- MAE:  {test_mae:.4f}")




# Kendall's τ

#Esto sirve para calcular si el modelo predice correctamente quien es mejor que quien y no solo la posicion exacta
def calculate_kendall_by_race(race_ids, y_true, y_pred):
    tau_scores = []
    for race_id in race_ids.unique():
        mask = race_ids == race_id
        y_race_true = y_true[mask]
        y_race_pred = y_pred[mask]
        if len(y_race_true) > 1:
            tau, _ = kendalltau(y_race_true, y_race_pred)
            if not np.isnan(tau):
                tau_scores.append(tau)
    return np.mean(tau_scores), np.std(tau_scores)

test_tau_mean, test_tau_std = calculate_kendall_by_race(
    test_race_ids, y_test, y_test_pred
)

print(f"\nKendall's τ (Test): {test_tau_mean:.4f} ± {test_tau_std:.4f}")




# 8. Descomposicion de influencia
print("\n" + "="*70)
print("8. DESCOMPOSICIÓN DE INFLUENCIA")
print("="*70)


coef_df = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'coefficient': best_ridge.coef_
})

coef_df['type'] = coef_df['feature'].apply(lambda x:
    'driver' if x.startswith('driver_') else
    'constructor' if x.startswith('constructor_') else
    'circuit' if x.startswith('circuit_') else
    'context'
)

variance_by_type = coef_df.groupby('type')['coefficient'].apply(
    lambda x: np.sum(x**2)
)

total_variance = variance_by_type.sum()
variance_pct = (variance_by_type / total_variance * 100).sort_values(ascending=False)

print("\nDESCOMPOSICIÓN DE VARIANZA:")
print("="*50)
for comp_type, pct in variance_pct.items():
    print(f"   {comp_type.upper():.<20} {pct:>6.2f}%")
print("="*50)


constructor_pct = variance_pct.get('constructor', 0)
driver_pct = variance_pct.get('driver', 0)
total_dc = constructor_pct + driver_pct
#Esto calcula el porcentaje **relativo** del piloto vs automovil

constructor_relative = constructor_pct / total_dc * 100
driver_relative = driver_pct / total_dc * 100


print(f"\nRESULTADO PRINCIPAL (Piloto vs. Automovil):")
print(f"   Automovil: {constructor_relative:.1f}%")
print(f"   Piloto:      {driver_relative:.1f}%")





# 9. Visualizaciones


plt.style.use('seaborn-v0_8-darkgrid')

# Gráfico 1: Pie Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
ax1.pie(variance_pct, labels=variance_pct.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
ax1.set_title('Descomposición de Varianza por Componente\n(Era Híbrida 2014-2024)',
              fontsize=13, weight='bold', pad=20)

driver_constructor = pd.Series({
    'Automovil': constructor_pct,
    'Piloto': driver_pct
})
ax2.pie(driver_constructor, labels=driver_constructor.index, autopct='%1.1f%%',
        colors=['#E74C3C', '#3498DB'], startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'})
ax2.set_title('Driver vs. Constructor\n(Solo componentes principales)',
              fontsize=13, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('influence_decomposition.png', dpi=300, bbox_inches='tight')


# Gráfico 2: Predicciones vs. Reales
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_test_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
ax.plot([1, 20], [1, 20], 'r--', lw=2, label='Predicción perfecta')
ax.set_xlabel('Posición Real', fontsize=12, weight='bold')
ax.set_ylabel('Posición Predicha', fontsize=12, weight='bold')
ax.set_title(f'Predicciones vs. Valores Reales (Test Set 2023-2024)\nKendall\'s τ = {test_tau_mean:.3f}',
             fontsize=13, weight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 21)
ax.set_ylim(0, 21)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')


# 10. Exportar resultados

results_summary = pd.DataFrame({
    'Metric': [
        'Model', 'Best Alpha', 'Train Period', 'Test Period',
        'Train Observations', 'Test Observations', 'Total Features',
        'Test RMSE', 'Test MAE', 'Test Kendall Tau',
        'Constructor Influence (%)', 'Driver Influence (%)'
    ],
    'Value': [
        'Ridge Regression', best_alpha, '2014-2022', '2023-2024',
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
        f'{np.sqrt(test_mse):.4f}', f'{test_mae:.4f}', f'{test_tau_mean:.4f}',
        f'{constructor_relative:.1f}', f'{driver_relative:.1f}'
    ]
})

results_summary.to_csv('model_results_summary.csv', index=False)
coef_df.to_csv('model_coefficients.csv', index=False)





# RESUMEN FINAL
print("\n" + "="*70)
print("✅ MODELO COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"\n🏁 RESULTADOS PRINCIPALES:")
print(f"   • Automovil: {constructor_relative:.1f}% de influencia")
print(f"   • Piloto: {driver_relative:.1f}% de influencia")
print(f"   • Kendall's τ (test): {test_tau_mean:.3f}")
print(f"   • RMSE (test): {np.sqrt(test_mse):.2f} posiciones")
print("\n" + "="*70)
