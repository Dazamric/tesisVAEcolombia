# %% Dependencias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import shap
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 150,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'font.family': 'sans-serif',
})
sns.set_style("whitegrid")

# %% Carga y preparación
RUTA_TABLA = r"C:\Users\usuario\Desktop\DataICFES\datos\output\tabla_analitica_VAE.csv"
RUTA_VAE_HLM = r"C:\Users\usuario\Desktop\DataICFES\datos\vae_hlm_por_ies.csv"
RUTA_VAE_ML = r"C:\Users\usuario\Desktop\DataICFES\datos\vae_ml_por_ies.csv"
MIN_ESTUDIANTES_IES = 30
EVALUAR_PROFUNDIDAD_RF = False
K_FOLDS = 5
RANDOM_STATE = 42

df_raw = pd.read_csv(RUTA_TABLA, low_memory=False)

# %%
df = df_raw.copy()

if 'spro_estu_agregado' in df.columns:
    df = df[df['spro_estu_agregado'] == 'S']

df['Y'] = pd.to_numeric(df['spro_punt_global'], errors='coerce')

df['s11_global'] = pd.to_numeric(df['s11_punt_global'], errors='coerce')

s11_modulos = {}
for col, name in [
    ('s11_punt_lectura_critica', 's11_lectura'),
    ('s11_punt_matematicas', 's11_matematicas'),
    ('s11_punt_c_naturales', 's11_cnaturales'),
    ('s11_punt_sociales_ciudadanas', 's11_sociales'),
    ('s11_punt_ingles', 's11_ingles'),
]:
    if col in df.columns:
        df[name] = pd.to_numeric(df[col], errors='coerce')
        s11_modulos[name] = col

if 'spro_estu_inse_individual' in df.columns:
    df['inse'] = pd.to_numeric(df['spro_estu_inse_individual'], errors='coerce')

cat_features = {}

for col in ['spro_estu_genero', 's11_estu_genero']:
    if col in df.columns:
        df['genero'] = (df[col] == 'M').astype(int)
        cat_features['genero'] = col
        break

if 'spro_fami_estratovivienda' in df.columns:
    estrato_map = {'Estrato 1': 1, 'Estrato 2': 2, 'Estrato 3': 3,
                   'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6}
    df['estrato'] = df['spro_fami_estratovivienda'].map(estrato_map)

if 'spro_fami_educacionmadre' in df.columns:
    edu_map = {'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
               'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4,
               'Técnica o tecnológica incompleta': 5, 'Técnica o tecnológica completa': 6,
               'Educación profesional incompleta': 7, 'Educación profesional completa': 8,
               'Postgrado': 9}
    df['edu_madre'] = df['spro_fami_educacionmadre'].map(edu_map)

if 'spro_fami_educacionpadre' in df.columns:
    df['edu_padre'] = df['spro_fami_educacionpadre'].map(edu_map)

for var_original, var_nueva in [
    ('spro_fami_tieneinternet', 'internet'),
    ('spro_fami_tienecomputador', 'computador'),
    ('spro_fami_tieneautomovil', 'automovil'),
    ('spro_fami_tienelavadora', 'lavadora'),
    ('spro_fami_tienemotocicleta', 'motocicleta'),
    ('spro_fami_tieneserviciotv', 'tv_cerrada'),
]:
    if var_original in df.columns:
        df[var_nueva] = (df[var_original] == 'Si').astype(int)

if 'spro_estu_horassemanatrabaja' in df.columns:
    horas_map = {'0': 0, 'Menos de 10 horas': 5, 'Entre 11 y 20 horas': 15,
                 'Entre 21 y 30 horas': 25, 'Más de 30 horas': 35}
    df['horas_trabajo'] = df['spro_estu_horassemanatrabaja'].map(horas_map)

if 'cohorte_spro' not in df.columns:
    df['cohorte_spro'] = df['periodo_sbpro'].astype(str).str[:4].astype(int)

if 'spro_inst_origen' in df.columns:
    df['ies_oficial'] = df['spro_inst_origen'].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith('OFICIAL') else 0)

if 'spro_inst_cod_institucion' in df.columns:
    df['ies_id'] = df['spro_inst_cod_institucion'].astype(str)
elif 'inst_cod_institucion' in df.columns:
    df['ies_id'] = df['inst_cod_institucion'].astype(str)

# %%
FEATURES_BASE = ['s11_global']
FEATURES_FULL = ['s11_global']

for name in s11_modulos:
    if name in df.columns and df[name].notna().mean() > 0.85:
        FEATURES_FULL.append(name)

sociodem_candidates = [
    'inse', 'genero', 'estrato', 'edu_madre', 'edu_padre',
    'internet', 'computador', 'automovil', 'lavadora',
    'motocicleta', 'tv_cerrada', 'horas_trabajo', 'cohorte_spro',
]
for feat in sociodem_candidates:
    if feat in df.columns and df[feat].notna().mean() > 0.85:
        FEATURES_FULL.append(feat)

print(f"Features BASE (solo S11): {FEATURES_BASE}")
print(f"\nFeatures FULL ({len(FEATURES_FULL)} variables):")
for f in FEATURES_FULL:
    miss = df[f].isna().mean() * 100
    print(f"  {f}: {miss:.1f}% miss")

# %%
cols_needed = FEATURES_FULL + ['Y', 'ies_id']
df_model = df.dropna(subset=cols_needed).copy()
print(f"\nDataset para modelado: {len(df_model):,} estudiantes")
print(f"   IES únicas: {df_model['ies_id'].nunique()}")
print(f"   Features: {len(FEATURES_FULL)}")

X_full = df_model[FEATURES_FULL].values.astype(np.float32)
X_base = df_model[FEATURES_BASE].values.astype(np.float32)
y = df_model['Y'].values.astype(np.float32)
groups = df_model['ies_id'].values

print(f"   X_full shape: {X_full.shape}")
print(f"   y shape: {y.shape}")

# %% Validación cruzada por IES
def evaluate_model_cv(model_class, model_params, X, y, groups, feature_names,
                      k=K_FOLDS, model_name="Modelo"):
    gkf = GroupKFold(n_splits=k)

    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    t_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)

        if isinstance(model, lgb.LGBMRegressor):
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_pred

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append({'fold': fold_idx + 1, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

        print(f"  Fold {fold_idx+1}/{k}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.4f}")

    elapsed = time.time() - t_start

    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    oof_mae = mean_absolute_error(y, oof_predictions)
    oof_r2 = r2_score(y, oof_predictions)

    print(f"\n  OOF Global: RMSE={oof_rmse:.3f}, MAE={oof_mae:.3f}, R²={oof_r2:.4f}")
    print(f"  Tiempo total: {elapsed:.1f}s")

    final_model = model_class(**model_params)
    if isinstance(final_model, lgb.LGBMRegressor):
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    else:
        final_model.fit(X, y)

    return {
        'name': model_name,
        'oof_predictions': oof_predictions,
        'oof_rmse': oof_rmse,
        'oof_mae': oof_mae,
        'oof_r2': oof_r2,
        'fold_metrics': pd.DataFrame(fold_metrics),
        'final_model': final_model,
        'elapsed': elapsed,
    }

# %% Regresión lineal
print("Regresión lineal, solo Saber 11")

result_lr_base = evaluate_model_cv(
    LinearRegression, {},
    X_base, y, groups, FEATURES_BASE,
    model_name="Reg. Lineal (solo S11)"
)

# %%
print("Regresión lineal, todas las variables")

result_lr_full = evaluate_model_cv(
    LinearRegression, {},
    X_full, y, groups, FEATURES_FULL,
    model_name="Reg. Lineal (full)"
)

# %% Elastic Net
print("Elastic Net")

result_enet = evaluate_model_cv(
    ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
    X_full, y, groups, FEATURES_FULL,
    model_name="Elastic Net"
)

# %% Random Forest
print("Random Forest")

result_rf = evaluate_model_cv(
    RandomForestRegressor,
    {'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 20,
     'n_jobs': -1, 'random_state': RANDOM_STATE},
    X_full, y, groups, FEATURES_FULL,
    model_name="Random Forest"
)

# %%
if EVALUAR_PROFUNDIDAD_RF:
    filas_rf = []
    for prof in [8, 10, 15, 20, None]:
        r_rf = evaluate_model_cv(
            RandomForestRegressor,
            {'n_estimators': 300, 'max_depth': prof, 'min_samples_leaf': 20,
             'n_jobs': -1, 'random_state': RANDOM_STATE},
            X_full, y, groups, FEATURES_FULL,
            model_name=f"RF (max_depth={prof})"
        )
        filas_rf.append({'max_depth': str(prof), 'RMSE (OOF)': r_rf['oof_rmse'],
                         'MAE (OOF)': r_rf['oof_mae'], 'R² (OOF)': r_rf['oof_r2']})
    tabla_rf = pd.DataFrame(filas_rf)
    print(tabla_rf.to_string(index=False, float_format='{:.4f}'.format))
    tabla_rf.to_csv('sensibilidad_rf_profundidad.csv', index=False)

# %% LightGBM
print("LightGBM")

result_lgbm = evaluate_model_cv(
    lgb.LGBMRegressor,
    {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 8,
     'num_leaves': 63, 'min_child_samples': 50,
     'subsample': 0.8, 'colsample_bytree': 0.8,
     'reg_alpha': 0.1, 'reg_lambda': 0.1,
     'n_jobs': -1, 'random_state': RANDOM_STATE, 'verbosity': -1},
    X_full, y, groups, FEATURES_FULL,
    model_name="LightGBM"
)

# %% Comparación de modelos
print("Comparación de modelos")

all_results = [result_lr_base, result_lr_full, result_enet, result_rf, result_lgbm]

comparison = pd.DataFrame([{
    'Modelo': r['name'],
    'RMSE (OOF)': r['oof_rmse'],
    'MAE (OOF)': r['oof_mae'],
    'R² (OOF)': r['oof_r2'],
    'Tiempo (s)': r['elapsed'],
} for r in all_results])

baseline_rmse = result_lr_base['oof_rmse']
comparison['Δ RMSE vs LR-S11'] = comparison['RMSE (OOF)'] - baseline_rmse
comparison['% mejora RMSE'] = ((baseline_rmse - comparison['RMSE (OOF)']) / baseline_rmse * 100)

print(comparison.to_string(index=False, float_format='{:.4f}'.format))

resumen_folds = pd.concat(
    [r['fold_metrics'].drop(columns='fold').agg(['mean', 'std']).T.assign(Modelo=r['name'])
     for r in all_results]
).reset_index().rename(columns={'index': 'Métrica'})
print(resumen_folds[['Modelo', 'Métrica', 'mean', 'std']].to_string(index=False, float_format='{:.4f}'.format))
resumen_folds.to_csv('metricas_por_fold.csv', index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models_names = [r['name'] for r in all_results]
r2_values = [r['oof_r2'] for r in all_results]
colors = ['#888780', '#888780', '#B4B2A9', '#378ADD', '#1D9E75']

bars = axes[0].barh(models_names, r2_values, color=colors, edgecolor='white', height=0.6)
axes[0].set_xlabel('R² (out-of-fold)')
axes[0].set_title('Capacidad predictiva: R² por modelo')
axes[0].set_xlim(0, max(r2_values) * 1.1)
for bar, val in zip(bars, r2_values):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=10)

rmse_values = [r['oof_rmse'] for r in all_results]
bars2 = axes[1].barh(models_names, rmse_values, color=colors, edgecolor='white', height=0.6)
axes[1].set_xlabel('RMSE (out-of-fold)')
axes[1].set_title('Error de predicción: RMSE por modelo')
for bar, val in zip(bars2, rmse_values):
    axes[1].text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10)

plt.tight_layout()
plt.savefig('fig_3_1_comparacion_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
best_result = min(all_results, key=lambda r: r['oof_rmse'])
print(f"\nMejor modelo: {best_result['name']} (RMSE={best_result['oof_rmse']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sample_idx = np.random.RandomState(42).choice(len(y), min(20000, len(y)), replace=False)
axes[0].hexbin(y[sample_idx], result_lr_base['oof_predictions'][sample_idx],
               gridsize=50, cmap='Reds', mincnt=1)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, linewidth=1)
axes[0].set_xlabel('Puntaje SPro observado')
axes[0].set_ylabel('Puntaje SPro predicho')
axes[0].set_title(f"Reg. Lineal solo S11 (R²={result_lr_base['oof_r2']:.4f})")

axes[1].hexbin(y[sample_idx], best_result['oof_predictions'][sample_idx],
               gridsize=50, cmap='Blues', mincnt=1)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, linewidth=1)
axes[1].set_xlabel('Puntaje SPro observado')
axes[1].set_ylabel('Puntaje SPro predicho')
axes[1].set_title(f"{best_result['name']} (R²={best_result['oof_r2']:.4f})")

plt.tight_layout()
plt.savefig('fig_3_2_predichos_vs_observados.png', dpi=300, bbox_inches='tight')
plt.show()

# %% SHAP
print("SHAP")

best_model = best_result['final_model']

shap_sample_size = min(10000, len(X_full))
shap_idx = np.random.RandomState(42).choice(len(X_full), shap_sample_size, replace=False)
X_shap = X_full[shap_idx]

print(f"Calculando SHAP values para {shap_sample_size:,} observaciones...")
t0 = time.time()

if isinstance(best_model, lgb.LGBMRegressor):
    explainer = shap.TreeExplainer(best_model)
elif isinstance(best_model, RandomForestRegressor):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.Explainer(best_model, X_shap)

shap_values = explainer.shap_values(X_shap)
print(f"SHAP calculado en {time.time() - t0:.1f}s")

# %%
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES_FULL,
                  show=False, max_display=20)
plt.title(f'SHAP: Importancia de variables ({best_result["name"]})')
plt.tight_layout()
plt.savefig('fig_3_3_shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES_FULL,
                  plot_type="bar", show=False, max_display=20)
plt.title(f'SHAP: Importancia promedio absoluta ({best_result["name"]})')
plt.tight_layout()
plt.savefig('fig_3_4_shap_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

s11_idx = FEATURES_FULL.index('s11_global')
axes[0].scatter(X_shap[:, s11_idx], shap_values[:, s11_idx],
                alpha=0.1, s=2, c='#378ADD')
axes[0].set_xlabel('Puntaje global Saber 11')
axes[0].set_ylabel('SHAP value (contribución a predicción SPro)')
axes[0].set_title('Efecto de S11 en la predicción (SHAP dependence)')
axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')

from statsmodels.nonparametric.smoothers_lowess import lowess
sorted_data = np.column_stack([X_shap[:, s11_idx], shap_values[:, s11_idx]])
sorted_data = sorted_data[sorted_data[:, 0].argsort()]
lowess_result = lowess(sorted_data[:, 1], sorted_data[:, 0], frac=0.3)
axes[0].plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2.5, label='LOWESS')
axes[0].legend()

if 'inse' in FEATURES_FULL:
    inse_idx = FEATURES_FULL.index('inse')
    axes[1].scatter(X_shap[:, inse_idx], shap_values[:, inse_idx],
                    alpha=0.1, s=2, c='#534AB7')
    axes[1].set_xlabel('INSE (Índice Socioeconómico)')
    axes[1].set_ylabel('SHAP value')
    axes[1].set_title('Efecto del INSE en la predicción')
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')

    sorted_inse = np.column_stack([X_shap[:, inse_idx], shap_values[:, inse_idx]])
    sorted_inse = sorted_inse[sorted_inse[:, 0].argsort()]
    lowess_inse = lowess(sorted_inse[:, 1], sorted_inse[:, 0], frac=0.3)
    axes[1].plot(lowess_inse[:, 0], lowess_inse[:, 1], 'r-', linewidth=2.5, label='LOWESS')
    axes[1].legend()

plt.tight_layout()
plt.savefig('fig_3_5_shap_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

# %% VAE-ML
print("VAE-ML")

df_model['Y_pred_ML'] = best_result['oof_predictions']
df_model['residuo_ML'] = df_model['Y'] - df_model['Y_pred_ML']

df_model['Y_pred_LR'] = result_lr_full['oof_predictions']
df_model['residuo_LR'] = df_model['Y'] - df_model['Y_pred_LR']

vae_ml = df_model.groupby('ies_id').agg(
    VAE_ML=('residuo_ML', 'mean'),
    VAE_LR=('residuo_LR', 'mean'),
    n_estudiantes=('Y', 'count'),
    spro_mean=('Y', 'mean'),
    s11_mean=('s11_global', 'mean'),
).reset_index()

for col in ['spro_inst_nombre_institucion', 'inst_nombre_institucion']:
    if col in df_model.columns:
        nombres = df_model.groupby('ies_id')[col].first().reset_index()
        nombres.columns = ['ies_id', 'nombre_ies']
        vae_ml = vae_ml.merge(nombres, on='ies_id', how='left')
        break

vae_ml = vae_ml.sort_values('VAE_ML', ascending=False)

vae_ml_min = vae_ml[vae_ml['n_estudiantes'] >= MIN_ESTUDIANTES_IES]
print(f"IES con al menos {MIN_ESTUDIANTES_IES} estudiantes: {len(vae_ml_min)} de {len(vae_ml)}")
print(f"Desviación estándar del VAE-ML (todas las IES): {vae_ml['VAE_ML'].std():.2f}; "
      f"(IES con n >= {MIN_ESTUDIANTES_IES}): {vae_ml_min['VAE_ML'].std():.2f}")

print(f"\nDistribución del VAE-ML:")
print(f"   Mean:  {vae_ml['VAE_ML'].mean():.2f}")
print(f"   Std:   {vae_ml['VAE_ML'].std():.2f}")
print(f"   Min:   {vae_ml['VAE_ML'].min():.2f}")
print(f"   Max:   {vae_ml['VAE_ML'].max():.2f}")

print(f"\nTop 10 IES por VAE-ML ")
for _, row in vae_ml.head(10).iterrows():
    nom = row.get('nombre_ies', row['ies_id'])
    if isinstance(nom, str) and len(nom) > 50:
        nom = nom[:50]
    print(f"  VAE={row['VAE_ML']:+.2f} | n={row['n_estudiantes']:,} | {nom}")

print(f"\nBottom 10 IES por VAE-ML ")
for _, row in vae_ml.tail(10).iterrows():
    nom = row.get('nombre_ies', row['ies_id'])
    if isinstance(nom, str) and len(nom) > 50:
        nom = nom[:50]
    print(f"  VAE={row['VAE_ML']:+.2f} | n={row['n_estudiantes']:,} | {nom}")

# %% Comparación VAE-HLM vs VAE-ML
print("Comparación VAE-HLM y VAE-ML")

vae_comparison = None
try:
    vae_hlm = pd.read_csv(RUTA_VAE_HLM)
    print(f"VAE-HLM cargado: {len(vae_hlm)} IES")

    vae_hlm['ies_id'] = vae_hlm['ies_id'].astype(str)
    vae_comparison = vae_ml.merge(vae_hlm[['ies_id', 'VAE_HLM']], on='ies_id', how='inner')
    print(f"IES en común: {len(vae_comparison)}")

    corr = vae_comparison['VAE_HLM'].corr(vae_comparison['VAE_ML'])
    print(f"\nCorrelación VAE-HLM vs VAE-ML: r = {corr:.4f}")

    spro_std = df_model['Y'].std()
    vae_comparison['VAE_ML_DE'] = vae_comparison['VAE_ML'] / spro_std

    corr_de = vae_comparison['VAE_HLM'].corr(vae_comparison['VAE_ML_DE'])
    print(f"  Correlación (VAE-ML en DE): r = {corr_de:.4f}")
except FileNotFoundError:
    print(f"No se encontró {RUTA_VAE_HLM}. Ejecutar antes el script HLM; se omite la comparación.")

# %%
if vae_comparison is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sizes = np.clip(vae_comparison['n_estudiantes'] / 20, 5, 200)
    axes[0].scatter(vae_comparison['VAE_HLM'], vae_comparison['VAE_ML_DE'],
                    s=sizes, alpha=0.5, c='#378ADD', edgecolors='white', linewidth=0.5)
    lim = max(abs(vae_comparison['VAE_HLM']).max(), abs(vae_comparison['VAE_ML_DE']).max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, linewidth=1)
    axes[0].set_xlabel('VAE-HLM (DE)')
    axes[0].set_ylabel('VAE-ML (DE)')
    axes[0].set_title(f'VAE-HLM vs VAE-ML (r = {corr_de:.3f})')
    axes[0].axhline(0, color='gray', linewidth=0.3)
    axes[0].axvline(0, color='gray', linewidth=0.3)

    vae_comparison['diff'] = vae_comparison['VAE_ML_DE'] - vae_comparison['VAE_HLM']
    axes[1].hist(vae_comparison['diff'], bins=40, color='#534AB7', edgecolor='white', alpha=0.7)
    axes[1].set_xlabel('Diferencia (VAE-ML - VAE-HLM) en DE')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de discrepancias')
    axes[1].axvline(0, color='red', linewidth=1, linestyle='--')

    vae_comparison['abs_diff'] = vae_comparison['diff'].abs()
    top_diff = vae_comparison.nlargest(10, 'abs_diff')

    axes[2].barh(range(len(top_diff)), top_diff['diff'].values,
                 color=['#E24B4A' if d < 0 else '#1D9E75' for d in top_diff['diff']],
                 height=0.6)
    labels = [str(row.get('nombre_ies', row['ies_id']))[:35] for _, row in top_diff.iterrows()]
    axes[2].set_yticks(range(len(top_diff)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel('Diferencia VAE-ML - VAE-HLM (DE)')
    axes[2].set_title('Top 10 IES con mayor discrepancia')
    axes[2].axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('fig_3_6_vae_hlm_vs_ml.png', dpi=300, bbox_inches='tight')
    plt.show()

# %%
os.makedirs(os.path.dirname(RUTA_VAE_ML), exist_ok=True)
vae_ml.to_csv(RUTA_VAE_ML, index=False)
vae_ml_min.to_csv('vae_ml_por_ies_n30.csv', index=False)

df_model[['Y', 'Y_pred_ML', 'residuo_ML', 'Y_pred_LR', 'residuo_LR', 'ies_id']].to_csv(
    'predicciones_oof_cap3.csv', index=False)
if vae_comparison is not None:
    vae_comparison.to_csv('vae_comparacion_hlm_vs_ml.csv', index=False)
