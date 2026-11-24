# %% Modelo jerárquico lineal
# %% Carga y preparación
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif',
})
sns.set_style("whitegrid")

# %%
RUTA_TABLA = r"C:\Users\usuario\Desktop\DataICFES\datos\output\tabla_analitica_VAE.csv"
RUTA_VAE_HLM = r"C:\Users\usuario\Desktop\DataICFES\datos\vae_hlm_por_ies.csv"

df_raw = pd.read_csv(RUTA_TABLA, low_memory=False)
print(f"Tabla cargada: {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

# %%
df = df_raw.copy()

if 'spro_estu_agregado' in df.columns:
    n_before = len(df)
    df = df[df['spro_estu_agregado'] == 'S']
    print(f"Filtro agregado SPro: {n_before:,}: {len(df):,} ({n_before - len(df):,} eliminados)")

score_cols_s11 = ['s11_punt_global', 's11_punt_lectura_critica', 's11_punt_matematicas',
                  's11_punt_c_naturales', 's11_punt_sociales_ciudadanas', 's11_punt_ingles']
score_cols_spro = ['spro_punt_global', 'spro_mod_lectura_critica_punt', 'spro_mod_razona_cuantitat_punt',
                   'spro_mod_competen_ciudada_punt', 'spro_mod_comuni_escrita_punt', 'spro_mod_ingles_punt']

for c in score_cols_s11 + score_cols_spro:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

n_before = len(df)
df = df.dropna(subset=['s11_punt_global', 'spro_punt_global'])
print(f"Filtro puntajes no nulos: {n_before:,}: {len(df):,} ({n_before - len(df):,} eliminados)")

if 'spro_estu_inse_individual' in df.columns:
    df['spro_estu_inse_individual'] = pd.to_numeric(df['spro_estu_inse_individual'], errors='coerce')

if 'cohorte_spro' not in df.columns:
    df['cohorte_spro'] = df['periodo_sbpro'].astype(str).str[:4].astype(int)

if 'spro_inst_cod_institucion' in df.columns:
    df['ies_id'] = df['spro_inst_cod_institucion'].astype(str)
elif 'inst_cod_institucion' in df.columns:
    df['ies_id'] = df['inst_cod_institucion'].astype(str)

if 'spro_estu_snies_prgmacademico' in df.columns:
    df['programa_id'] = df['spro_estu_snies_prgmacademico'].astype(str)

for cohorte in df['cohorte_spro'].unique():
    mask = df['cohorte_spro'] == cohorte
    col = 's11_punt_global'
    df.loc[mask, 's11_global_z'] = (df.loc[mask, col] - df.loc[mask, col].mean()) / df.loc[mask, col].std()
    col = 'spro_punt_global'
    df.loc[mask, 'spro_global_z'] = (df.loc[mask, col] - df.loc[mask, col].mean()) / df.loc[mask, col].std()
    for col in ['spro_mod_lectura_critica_punt', 'spro_mod_razona_cuantitat_punt']:
        if col in df.columns:
            zname = col.replace('spro_mod_', '').replace('_punt', '_z')
            df.loc[mask, zname] = (df.loc[mask, col] - df.loc[mask, col].mean()) / df.loc[mask, col].std()

print(f"\nDataset preparado: {len(df):,} estudiantes")
print(f"   IES únicas: {df['ies_id'].nunique()}")
print(f"   Cohortes: {sorted(df['cohorte_spro'].unique())}")

# %% Estadísticas descriptivas
print("Estadísticas descriptivas")
desc_vars = {
    's11_punt_global': 'Puntaje global Saber 11',
    'spro_punt_global': 'Puntaje global Saber Pro',
}

for col, label in [
    ('spro_mod_lectura_critica_punt', 'SPro: Lectura Crítica'),
    ('spro_mod_razona_cuantitat_punt', 'SPro: Razonamiento Cuantitativo'),
    ('spro_mod_competen_ciudada_punt', 'SPro: Competencias Ciudadanas'),
    ('spro_mod_comuni_escrita_punt', 'SPro: Comunicación Escrita'),
    ('spro_mod_ingles_punt', 'SPro: Inglés'),
]:
    if col in df.columns:
        desc_vars[col] = label

results = []
for col, label in desc_vars.items():
    s = df[col].dropna()
    results.append({
        'Variable': label,
        'N': len(s),
        'Media': s.mean(),
        'DE': s.std(),
        'Min': s.min(),
        'Q1': s.quantile(0.25),
        'Mediana': s.median(),
        'Q3': s.quantile(0.75),
        'Max': s.max(),
    })

tabla_desc = pd.DataFrame(results)
print(tabla_desc.to_string(index=False, float_format='{:.1f}'.format))

# %%
ies_counts = df.groupby('ies_id').size().reset_index(name='n_estudiantes')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(ies_counts['n_estudiantes'], bins=50, color='#378ADD', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Número de estudiantes por IES')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución del tamaño de IES')
axes[0].axvline(ies_counts['n_estudiantes'].median(), color='#E24B4A', linestyle='--',
                label=f"Mediana: {ies_counts['n_estudiantes'].median():.0f}")
axes[0].legend()

axes[1].boxplot(ies_counts['n_estudiantes'], vert=True)
axes[1].set_ylabel('Número de estudiantes')
axes[1].set_title('Distribución del tamaño de IES (Box Plot)')

plt.tight_layout()
plt.savefig('fig_2_1_distribucion_ies.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nEstadísticas de tamaño de IES:")
print(ies_counts['n_estudiantes'].describe().to_string())

# %%
sample = df.sample(min(20000, len(df)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hexbin(sample['s11_punt_global'], sample['spro_punt_global'],
               gridsize=50, cmap='Blues', mincnt=1)
axes[0].set_xlabel('Puntaje Global Saber 11')
axes[0].set_ylabel('Puntaje Global Saber Pro')
axes[0].set_title(f'Saber 11 y Saber Pro (r = {df["s11_punt_global"].corr(df["spro_punt_global"]):.4f})')
z = np.polyfit(sample['s11_punt_global'].dropna(), sample['spro_punt_global'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(sample['s11_punt_global'].min(), sample['s11_punt_global'].max(), 100)
axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7)

from statsmodels.nonparametric.smoothers_lowess import lowess
sample_sorted = sample.dropna(subset=['s11_punt_global', 'spro_punt_global']).sort_values('s11_punt_global')
lowess_result = lowess(sample_sorted['spro_punt_global'], sample_sorted['s11_punt_global'], frac=0.3)
axes[1].hexbin(sample['s11_punt_global'], sample['spro_punt_global'],
               gridsize=50, cmap='Blues', mincnt=1)
axes[1].plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2.5, label='LOWESS')
axes[1].plot(x_line, p(x_line), '--', color='#888780', linewidth=1.5, alpha=0.7, label='Lineal')
axes[1].set_xlabel('Puntaje Global Saber 11')
axes[1].set_ylabel('Puntaje Global Saber Pro')
axes[1].set_title('Detección de no linealidad: LOWESS vs Lineal')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_2_2_scatter_s11_spro.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("Descriptivos por cohorte")

by_cohorte = df.groupby('cohorte_spro').agg(
    N=('spro_punt_global', 'count'),
    S11_mean=('s11_punt_global', 'mean'),
    S11_std=('s11_punt_global', 'std'),
    SPro_mean=('spro_punt_global', 'mean'),
    SPro_std=('spro_punt_global', 'std'),
    n_IES=('ies_id', 'nunique'),
    corr_s11_spro=('s11_punt_global', lambda x: x.corr(df.loc[x.index, 'spro_punt_global'])),
).round(2)

print(by_cohorte.to_string())

# %% Modelo nulo
print("Modelo nulo")

modelo_nulo = smf.mixedlm(
    "spro_global_z ~ 1",
    data=df,
    groups=df["ies_id"],
)
resultado_nulo = modelo_nulo.fit(reml=True)

print(resultado_nulo.summary())

var_ies = resultado_nulo.cov_re.iloc[0, 0]
var_residual = resultado_nulo.scale
icc = var_ies / (var_ies + var_residual)

print("Resultados del modelo nulo")
print(f"  Varianza entre IES (τ₀₀): {var_ies:.4f}")
print(f"  Varianza residual (σ²):   {var_residual:.4f}")
print(f"  ICC = τ₀₀ / (τ₀₀ + σ²):  {icc:.4f} ({icc*100:.1f}%)")

# %% Modelo 1
print("Modelo 1")

modelo_1 = smf.mixedlm(
    "spro_global_z ~ s11_global_z",
    data=df,
    groups=df["ies_id"],
)
resultado_1 = modelo_1.fit(reml=True)

print(resultado_1.summary())

var_ies_1 = resultado_1.cov_re.iloc[0, 0]
var_res_1 = resultado_1.scale
icc_1 = var_ies_1 / (var_ies_1 + var_res_1)

print("Resultados del modelo 1")
print(f"  Coeficiente S11 global (β₁): {resultado_1.fe_params['s11_global_z']:.4f}")
print(f"  Varianza entre IES (τ₀₀):    {var_ies_1:.4f}")
print(f"  Varianza residual (σ²):       {var_res_1:.4f}")
print(f"  ICC condicional:              {icc_1:.4f} ({icc_1*100:.1f}%)")

# %% Modelo 2
print("Preparando covariables sociodemográficas...\n")

df['cohorte_2019'] = (df['cohorte_spro'] == 2019).astype(int)
df['cohorte_2022'] = (df['cohorte_spro'] == 2022).astype(int)
df['cohorte_2023'] = (df['cohorte_spro'] == 2023).astype(int)

if 'spro_estu_genero' in df.columns:
    df['genero_M'] = (df['spro_estu_genero'] == 'M').astype(int)
elif 'estu_genero' in df.columns:
    df['genero_M'] = (df['estu_genero'] == 'M').astype(int)

if 'spro_estu_inse_individual' in df.columns:
    inse = df['spro_estu_inse_individual']
    df['inse_z'] = (inse - inse.mean()) / inse.std()
    print(f"  INSE: mean={inse.mean():.1f}, std={inse.std():.1f}, miss={inse.isna().mean()*100:.1f}%")

if 'spro_fami_educacionmadre' in df.columns:
    edu_order = {
        'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
        'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4,
        'Técnica o tecnológica incompleta': 5, 'Técnica o tecnológica completa': 6,
        'Educación profesional incompleta': 7, 'Educación profesional completa': 8,
        'Postgrado': 9, 'No sabe': np.nan, 'No Aplica': np.nan,
    }
    df['edu_madre_ord'] = df['spro_fami_educacionmadre'].map(edu_order)
    print(f"  Educación madre: miss={df['edu_madre_ord'].isna().mean()*100:.1f}%")

if 'spro_fami_estratovivienda' in df.columns:
    estrato_map = {
        'Estrato 1': 1, 'Estrato 2': 2, 'Estrato 3': 3,
        'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6,
        'Sin Estrato': np.nan, 'Sin estrato': np.nan,
        'Vive en una zona rural donde no hay estratificación socioeconómica': np.nan,
    }
    df['estrato_num'] = df['spro_fami_estratovivienda'].map(estrato_map)
    print(f"  Estrato: miss={df['estrato_num'].isna().mean()*100:.1f}%")

if 'spro_fami_tieneinternet' in df.columns:
    df['tiene_internet'] = (df['spro_fami_tieneinternet'] == 'Si').astype(int)

if 'spro_fami_tienecomputador' in df.columns:
    df['tiene_computador'] = (df['spro_fami_tienecomputador'] == 'Si').astype(int)

if 'spro_inst_origen' in df.columns:
    df['ies_oficial'] = df['spro_inst_origen'].str.contains('OFICIAL', na=False).astype(int)
    df['ies_oficial'] = df['spro_inst_origen'].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith('OFICIAL') else 0
    )

print(f"\nCovariables preparadas")

# %%
print("Modelo 2")

formula_parts = ['spro_global_z ~ s11_global_z']

covariables = {
    'genero_M': 'Género (M)',
    'inse_z': 'INSE (z)',
    'edu_madre_ord': 'Educación madre (ordinal)',
    'estrato_num': 'Estrato',
    'tiene_internet': 'Internet en hogar',
    'tiene_computador': 'Computador en hogar',
    'cohorte_2019': 'Cohorte 2019',
    'cohorte_2022': 'Cohorte 2022',
    'cohorte_2023': 'Cohorte 2023',
}

covs_included = []
for var, label in covariables.items():
    if var in df.columns:
        miss = df[var].isna().mean()
        if miss < 0.15:
            formula_parts.append(var)
            covs_included.append((var, label, miss))
            print(f"   {label} ({var}): {miss*100:.1f}% faltantes, incluida")
        else:
            print(f"   {label} ({var}): {miss*100:.1f}% faltantes, excluida")

formula_str = ' + '.join(formula_parts)
print(f"\nFórmula: {formula_str} + (1 | IES)")

covs_cols = [v for v, _, _ in covs_included]
df_modelo = df.dropna(subset=['spro_global_z', 's11_global_z'] + covs_cols)
print(f"\nN después de eliminar missings: {len(df_modelo):,}")

print("\nEstimando modelo 2... (puede tomar 3-8 minutos)")

modelo_2 = smf.mixedlm(
    formula_str,
    data=df_modelo,
    groups=df_modelo["ies_id"],
)
resultado_2 = modelo_2.fit(reml=True)

print(resultado_2.summary())

var_ies_2 = resultado_2.cov_re.iloc[0, 0]
var_res_2 = resultado_2.scale
icc_2 = var_ies_2 / (var_ies_2 + var_res_2)

print("Resultados del modelo 2")
print(f"  Varianza entre IES (τ₀₀):  {var_ies_2:.4f}")
print(f"  Varianza residual (σ²):     {var_res_2:.4f}")
print(f"  ICC condicional:            {icc_2:.4f} ({icc_2*100:.1f}%)")

# %% Comparación de modelos
print("Comparación de modelos")

tabla_modelos = pd.DataFrame({
    'Modelo': ['Nulo', 'M1: Solo S11', 'M2: S11 + Sociodem.'],
    'τ₀₀ (var. entre IES)': [var_ies, var_ies_1, var_ies_2],
    'σ² (var. residual)': [var_residual, var_res_1, var_res_2],
    'ICC': [icc, icc_1, icc_2],
    'ICC (%)': [f"{icc*100:.1f}%", f"{icc_1*100:.1f}%", f"{icc_2*100:.1f}%"],
})

for i, res in enumerate([resultado_nulo, resultado_1, resultado_2]):
    try:
        tabla_modelos.loc[i, 'AIC'] = res.aic
        tabla_modelos.loc[i, 'BIC'] = res.bic
    except:
        pass

print(tabla_modelos.to_string(index=False))

# %%
print("Efectos fijos del modelo 2")

fe_params = resultado_2.fe_params
fe_bse = resultado_2.bse_fe
fe_ci = resultado_2.conf_int().loc[fe_params.index]

coefs = pd.DataFrame({
    'Variable': fe_params.index,
    'β': fe_params.values,
    'SE': fe_bse.values,
    'z': (fe_params / fe_bse).values,
    'p-valor': [resultado_2.pvalues[v] for v in fe_params.index],
    'IC 95% inf': fe_ci.iloc[:, 0].values,
    'IC 95% sup': fe_ci.iloc[:, 1].values,
})
coefs['Sig.'] = coefs['p-valor'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
)
print(coefs.to_string(index=False, float_format='{:.4f}'.format))

# %% VAE por IES
print("VAE por IES")

blups = resultado_2.random_effects
vae_hlm = pd.DataFrame({
    'ies_id': list(blups.keys()),
    'VAE_HLM': [v['Group'] for v in blups.values()],
})

ies_info = df_modelo.groupby('ies_id').agg(
    n_estudiantes=('spro_punt_global', 'count'),
).reset_index()

if 'spro_inst_nombre_institucion' in df_modelo.columns:
    nombres = df_modelo.groupby('ies_id')['spro_inst_nombre_institucion'].first().reset_index()
    nombres.columns = ['ies_id', 'nombre_ies']
    ies_info = ies_info.merge(nombres, on='ies_id', how='left')

vae_hlm = vae_hlm.merge(ies_info, on='ies_id', how='left')
vae_hlm = vae_hlm.sort_values('VAE_HLM', ascending=False)

print(f"\nDistribución del VAE-HLM:")
print(f"   Mean:   {vae_hlm['VAE_HLM'].mean():.4f}")
print(f"   Std:    {vae_hlm['VAE_HLM'].std():.4f}")
print(f"   Min:    {vae_hlm['VAE_HLM'].min():.4f}")
print(f"   Max:    {vae_hlm['VAE_HLM'].max():.4f}")
print(f"   IES con VAE > 0: {(vae_hlm['VAE_HLM'] > 0).sum()} de {len(vae_hlm)}")

print(f"\nTop 10 IES con mayor VAE (en DE de puntaje SPro)")
top10 = vae_hlm.head(10)
for _, row in top10.iterrows():
    nom = row.get('nombre_ies', row['ies_id'])
    if isinstance(nom, str) and len(nom) > 50:
        nom = nom[:50]
    print(f"  VAE={row['VAE_HLM']:+.3f} | n={row['n_estudiantes']:,} | {nom}")

print(f"\nBottom 10 IES con menor VAE")
bottom10 = vae_hlm.tail(10)
for _, row in bottom10.iterrows():
    nom = row.get('nombre_ies', row['ies_id'])
    if isinstance(nom, str) and len(nom) > 50:
        nom = nom[:50]
    print(f"  VAE={row['VAE_HLM']:+.3f} | n={row['n_estudiantes']:,} | {nom}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vae_sorted = vae_hlm.sort_values('VAE_HLM').reset_index(drop=True)
colors = ['#E24B4A' if v < 0 else '#378ADD' for v in vae_sorted['VAE_HLM']]
axes[0].barh(range(len(vae_sorted)), vae_sorted['VAE_HLM'], color=colors, alpha=0.6, height=1)
axes[0].axvline(0, color='black', linewidth=0.5)
axes[0].set_xlabel('VAE (desviaciones estándar)')
axes[0].set_ylabel('IES (ordenadas por VAE)')
axes[0].set_title('Ranking de IES por VAE-HLM (Modelo 2)')
axes[0].set_yticks([])

axes[1].scatter(vae_hlm['n_estudiantes'], vae_hlm['VAE_HLM'],
                alpha=0.5, s=20, c='#378ADD')
axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1].set_xlabel('Número de estudiantes en la IES')
axes[1].set_ylabel('VAE-HLM (DE)')
axes[1].set_title('VAE vs Tamaño de IES')
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('fig_2_3_vae_hlm.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
os.makedirs(os.path.dirname(RUTA_VAE_HLM), exist_ok=True)
vae_hlm.to_csv(RUTA_VAE_HLM, index=False)
print(f"VAE-HLM guardado en {RUTA_VAE_HLM} ({len(vae_hlm)} IES)")

# %% Diagnósticos
residuos = resultado_2.resid

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(residuos, bins=80, color='#378ADD', edgecolor='white', density=True, alpha=0.7)
x_norm = np.linspace(residuos.min(), residuos.max(), 100)
axes[0].plot(x_norm, stats.norm.pdf(x_norm, residuos.mean(), residuos.std()),
             'r-', linewidth=2, label='Normal teórica')
axes[0].set_xlabel('Residuo')
axes[0].set_ylabel('Densidad')
axes[0].set_title('Distribución de residuos (Modelo 2)')
axes[0].legend()

stats.probplot(residuos.dropna(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot de residuos')

predichos = resultado_2.fittedvalues
axes[2].scatter(predichos, residuos, alpha=0.05, s=1, c='#378ADD')
axes[2].axhline(0, color='red', linewidth=0.5)
axes[2].set_xlabel('Valores predichos')
axes[2].set_ylabel('Residuos')
axes[2].set_title('Residuos vs Predichos')

plt.tight_layout()
plt.savefig('fig_2_4_diagnosticos_hlm.png', dpi=300, bbox_inches='tight')
plt.show()

jb_stat, jb_p = stats.jarque_bera(residuos.dropna())
print(f"\nTest de normalidad (Jarque-Bera):")
print(f"   Estadístico: {jb_stat:.1f}")
print(f"   p-valor: {jb_p:.2e}")
print(f"   N: {len(residuos):,}")

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vae_hlm['VAE_HLM'], bins=40, color='#534AB7', edgecolor='white', alpha=0.7, density=True)
x_norm = np.linspace(vae_hlm['VAE_HLM'].min(), vae_hlm['VAE_HLM'].max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, vae_hlm['VAE_HLM'].mean(), vae_hlm['VAE_HLM'].std()),
        'r-', linewidth=2, label='Normal teórica')
ax.set_xlabel('VAE-HLM (DE)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución del VAE-HLM entre IES (BLUPs)')
ax.legend()
plt.savefig('fig_2_5_distribucion_blups.png', dpi=300, bbox_inches='tight')
plt.show()

shapiro_stat, shapiro_p = stats.shapiro(vae_hlm['VAE_HLM'].dropna().values[:min(5000, len(vae_hlm))])
print(f"\nNormalidad de BLUPs (Shapiro-Wilk): W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
