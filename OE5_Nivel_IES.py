# %% Comparación a nivel de IES
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
})

# %% Configuración
RUTA_VA_INBC = "salida_oe5/va_espejo_por_inbc_2023_2024.csv"
RUTA_VAE_ML = r"C:\Users\usuario\Desktop\DataICFES\datos\vae_ml_por_ies.csv"
RUTA_SALIDA = "salida_oe5"
os.makedirs(RUTA_SALIDA, exist_ok=True)

COL_S11 = 's11_mean'
COL_BRUTO = 'spro_mean'

MEDIDAS = {
    'Modelo-espejo': 'espejo',
    'VAE-ML': 'VAE_ML',
    'VAE-LR': 'VAE_LR',
    'Puntaje bruto': COL_BRUTO,
}

PARES = [
    ('Modelo-espejo', 'VAE-ML'),
    ('Modelo-espejo', 'VAE-LR'),
    ('VAE-ML', 'VAE-LR'),
    ('Modelo-espejo', 'Puntaje bruto'),
    ('VAE-ML', 'Puntaje bruto'),
    ('VAE-LR', 'Puntaje bruto'),
]

# %% Funciones de agregación
def media_ponderada(d, x, w, claves):
    t = (d.assign(_xw=d[x] * d[w])
          .groupby(claves)
          .agg(_xw=('_xw', 'sum'), _w=(w, 'sum')))
    return t['_xw'] / t['_w']


def agregar_espejo_a_ies(va):
    g = va.groupby(['nbc_norm', 'competencia'])['VA_espejo']
    va = va.copy()
    va['z'] = (va['VA_espejo'] - g.transform('mean')) / g.transform('std')
    va = va.dropna(subset=['z'])

    inbc = (va.groupby(['inbc_id', 'ies_id'])
              .agg(z=('z', 'mean'), n=('n_estudiantes', 'mean'))
              .reset_index())

    ies = media_ponderada(inbc, 'z', 'n', 'ies_id').rename('espejo')
    n_inbc = inbc.groupby('ies_id').size().rename('n_inbc')

    por_comp = media_ponderada(va, 'z', 'n_estudiantes', ['ies_id', 'competencia']).unstack()
    por_comp.columns = ['espejo_' + c for c in por_comp.columns]

    out = pd.concat([ies, n_inbc, por_comp], axis=1).reset_index()
    out['ies_id'] = out['ies_id'].astype(str)
    return out


def unir_con_ml(espejo, ml):
    ml = ml.copy()
    ml['ies_id'] = ml['ies_id'].astype(str)
    print(f"IES con VA espejo: {espejo['ies_id'].nunique()}")
    print(f"IES en vae_ml_por_ies.csv: {ml['ies_id'].nunique()}")
    d = espejo.merge(ml[['ies_id', 'VAE_ML', 'VAE_LR', 'n_estudiantes', COL_BRUTO, COL_S11]],
                     on='ies_id', how='inner')
    print(f"IES en ambas: {len(d)}")
    sin_ml = sorted(set(espejo['ies_id']) - set(ml['ies_id']))
    if sin_ml:
        print(f"IES con espejo y sin VAE_ML: {sin_ml}")
    d = d.dropna(subset=['espejo', 'VAE_ML', 'VAE_LR', COL_BRUTO, COL_S11])
    print(f"IES completas para la comparación: {len(d)}")
    return d


def tercil(s):
    return pd.qcut(s.rank(method='first'), 3, labels=[1, 2, 3]).astype(int)

# %% Concordancia, sesgo de composición y descomposición
def concordancia(d):
    filas = []
    for a, b in PARES:
        x, y = d[MEDIDAS[a]], d[MEDIDAS[b]]
        rho, p_rho = stats.spearmanr(x, y)
        tau, p_tau = stats.kendalltau(x, y)
        ta, tb = tercil(x), tercil(y)
        filas.append({
            'medida_A': a, 'medida_B': b, 'n_IES': len(d),
            'spearman': rho, 'p_spearman': p_rho,
            'kendall': tau, 'p_kendall': p_tau,
            'kappa_ponderado': cohen_kappa_score(ta, tb, weights='quadratic'),
            'pct_cambia_tercil': (ta != tb).mean(),
        })
    return pd.DataFrame(filas)


def sesgo_composicion(d):
    medidas = {
        'Puntaje bruto promedio de Saber Pro': COL_BRUTO,
        'Modelo-espejo, inglés': 'espejo_ingles',
        'Modelo-espejo (agregado)': 'espejo',
        'VAE-LR': 'VAE_LR',
        'VAE-ML': 'VAE_ML',
        'Modelo-espejo, razonamiento cuantitativo': 'espejo_razona_cuantitativo',
        'Modelo-espejo, lectura crítica': 'espejo_lectura_critica',
    }
    filas = []
    for etq, col in medidas.items():
        if col not in d.columns:
            continue
        g = d.dropna(subset=[col, COL_S11])
        r, p = stats.pearsonr(g[COL_S11], g[col])
        filas.append({'medida': etq, 'n_IES': len(g), 'r_con_S11': r, 'R2': r ** 2, 'p': p})
    return pd.DataFrame(filas).sort_values('R2', ascending=False).reset_index(drop=True)


def descomposicion(d):
    rk = lambda s: s.rank(pct=True)
    d = d.copy()
    d['desp_ML'] = rk(d['VAE_ML']) - rk(d['espejo'])
    d['desp_LR'] = rk(d['VAE_LR']) - rk(d['espejo'])
    d['desp_ML_menos_LR'] = d['desp_ML'] - d['desp_LR']

    rho, p_rho = stats.spearmanr(d['desp_ML'], d['desp_LR'])
    filas = [{
        'prueba': 'Spearman entre desp_ML y desp_LR',
        'estimacion': rho, 'p': p_rho, 'n_IES': len(d),
    }]
    for etq, col in [('desp_ML', 'desp_ML'), ('desp_LR', 'desp_LR'),
                     ('desp_ML_menos_LR', 'desp_ML_menos_LR')]:
        r, p = stats.pearsonr(d[COL_S11], d[col])
        filas.append({'prueba': f'Pearson entre {etq} y S11 promedio',
                      'estimacion': r, 'p': p, 'n_IES': len(d)})
    return pd.DataFrame(filas), d

# %% Figuras
def figura_sesgo(sesgo):
    s = sesgo.sort_values('R2')
    colores = ['#E24B4A' if 'bruto' in m else '#378ADD' for m in s['medida']]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    barras = ax.barh(s['medida'], s['R2'], color=colores)
    for b, v in zip(barras, s['R2']):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('R² al regresar la medida sobre el Saber 11 promedio de entrada')
    ax.set_title(f'Sesgo de composición por medida (nivel IES, n = {int(s["n_IES"].max())})')
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/fig_5_4_sesgo_composicion_ies.png', dpi=300, bbox_inches='tight')
    plt.close()


def figura_dispersion(d, rho):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(d['espejo'], d['VAE_ML'], c=d[COL_S11], cmap='viridis',
                    s=24, alpha=0.8, edgecolors='white', linewidth=0.4)
    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0, color='gray', linewidth=0.6)
    ax.set_xlabel('VA modelo-espejo (z agregado a IES)')
    ax.set_ylabel('VAE propuesta (ML), puntos de Saber Pro')
    ax.set_title(f'Modelo-espejo frente a la propuesta basada en ML\n'
                 f'Spearman = {rho:.3f} (n = {len(d)} IES)')
    cb = plt.colorbar(sc)
    cb.set_label('Saber 11 promedio de entrada')
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/fig_5_5_espejo_vs_ml_ies.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Ejecución
def ejecutar():
    va = pd.read_csv(RUTA_VA_INBC, dtype={'ies_id': str, 'inbc_id': str})
    ml = pd.read_csv(RUTA_VAE_ML, dtype={'ies_id': str})

    espejo = agregar_espejo_a_ies(va)
    d = unir_con_ml(espejo, ml)
    d.to_csv(f'{RUTA_SALIDA}/ies_tabla_comparacion.csv', index=False)

    conc = concordancia(d)
    conc.to_csv(f'{RUTA_SALIDA}/ies_concordancia.csv', index=False)
    print('\nconcordancia entre medidas')
    print(conc.round(3).to_string(index=False))

    sesgo = sesgo_composicion(d)
    sesgo.to_csv(f'{RUTA_SALIDA}/ies_sesgo_composicion.csv', index=False)
    print('\nsesgo de composición')
    print(sesgo.round(3).to_string(index=False))

    desc, d_desp = descomposicion(d)
    desc.to_csv(f'{RUTA_SALIDA}/ies_descomposicion.csv', index=False)
    d_desp.to_csv(f'{RUTA_SALIDA}/ies_desplazamientos.csv', index=False)
    print('\ndescomposición del desplazamiento de rango')
    print(desc.round(4).to_string(index=False))

    rho_ml = conc.loc[(conc['medida_A'] == 'Modelo-espejo') & (conc['medida_B'] == 'VAE-ML'), 'spearman'].iloc[0]
    figura_sesgo(sesgo)
    figura_dispersion(d, rho_ml)
    print(f'\nresultados guardados en {RUTA_SALIDA}/')

# %%
if __name__ == '__main__':
    ejecutar()
