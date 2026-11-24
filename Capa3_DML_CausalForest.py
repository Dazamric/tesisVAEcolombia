# %% Capa 3: DML y Causal Forests
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, roc_auc_score
import lightgbm as lgb
import doubleml as dml
from econml.dml import CausalForestDML

warnings.filterwarnings('ignore')

# %% Configuración
RUTA_TABLA = r"C:\Users\usuario\Desktop\DataICFES\datos\output\tabla_analitica_VAE.csv"
RUTA_SALIDA = "salida_capa3"
RANDOM_STATE = 42

N_FOLDS = 5
N_REP = 1
TRIMMING = 0.01

RESULTADOS = {
    'global': 'spro_punt_global',
    'lectura_critica': 'spro_mod_lectura_critica_punt',
    'razona_cuantitativo': 'spro_mod_razona_cuantitat_punt',
    'ingles': 'spro_mod_ingles_punt',
}
RESULTADO_CF = 'global'

ESPEC_PRINCIPAL = 'pre_ingreso'
CORRER_ROBUSTEZ_ML19 = True

CF_N_MAX = 200_000
CF_N_ESTIMATORS = 1000
CF_MIN_LEAF = 50
CF_MODULADORES = ['inse_s11', 's11_global', 'genero', 'estrato', 'edu_madre', 's11_ingles']

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=8,
                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                   n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
os.makedirs(RUTA_SALIDA, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11,
                     'font.family': 'sans-serif'})
def reg():
    return lgb.LGBMRegressor(**LGBM_PARAMS)
def clf():
    return lgb.LGBMClassifier(**LGBM_PARAMS)

# %% Carga y preparación
EDU_MAP = {'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
           'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4,
           'Técnica o tecnológica incompleta': 5, 'Técnica o tecnológica completa': 6,
           'Educación profesional incompleta': 7, 'Educación profesional completa': 8,
           'Postgrado': 9}
ESTRATO_MAP = {f'Estrato {i}': i for i in range(1, 7)}
HORAS_MAP = {'0': 0, 'Menos de 10 horas': 5, 'Entre 11 y 20 horas': 15,
             'Entre 21 y 30 horas': 25, 'Más de 30 horas': 35}
MODULOS_S11 = [('s11_punt_lectura_critica', 's11_lectura'), ('s11_punt_matematicas', 's11_matematicas'),
               ('s11_punt_c_naturales', 's11_cnaturales'), ('s11_punt_sociales_ciudadanas', 's11_sociales'),
               ('s11_punt_ingles', 's11_ingles')]

def preparar(df_raw):
    atr = []
    def registrar(etapa, d):
        atr.append({'etapa': etapa, 'estudiantes': len(d),
                    'IES': d['spro_inst_cod_institucion'].nunique()})

    df = df_raw.copy()
    registrar('0. Tabla analítica', df)
    if 'spro_estu_agregado' in df.columns:
        df = df[df['spro_estu_agregado'] == 'S']
        registrar('1. Marca de agregado en Saber Pro', df)

    for c in ['spro_punt_global', 's11_punt_global', 's11_estu_inse_individual',
              'spro_estu_inse_individual'] + [m for m, _ in MODULOS_S11] + list(RESULTADOS.values()):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['spro_punt_global', 's11_punt_global'])
    registrar('2. Puntaje global no nulo en ambas pruebas', df)

    if 'periodo_sb11' in df.columns:
        per = pd.to_numeric(df['periodo_sb11'].astype(str).str[:5], errors='coerce')
        df = df[per >= 20142]
        registrar('3. Saber 11 desde 2014-2', df)

    origen = df['spro_inst_origen'].astype(str).str.upper().str.strip()
    df = df[origen.notna() & (origen != 'NAN') & (origen != '')]
    df['D'] = df['spro_inst_origen'].astype(str).str.upper().str.startswith('OFICIAL').astype(int)
    registrar('4. Sector de la IES conocido', df)

    df['ies_id'] = df['spro_inst_cod_institucion'].astype(str)
    df['cluster'] = pd.factorize(df['ies_id'])[0]

    df['s11_global'] = df['s11_punt_global']
    for col, name in MODULOS_S11:
        df[name] = df[col] if col in df.columns else np.nan
    df['inse_s11'] = df['s11_estu_inse_individual'] if 's11_estu_inse_individual' in df.columns else np.nan
    df['inse'] = df['spro_estu_inse_individual'] if 'spro_estu_inse_individual' in df.columns else np.nan
    df['genero'] = (df['spro_estu_genero'] == 'M').astype(int)
    df['estrato'] = df['spro_fami_estratovivienda'].map(ESTRATO_MAP)
    df['edu_madre'] = df['spro_fami_educacionmadre'].map(EDU_MAP)
    df['edu_padre'] = df['spro_fami_educacionpadre'].map(EDU_MAP)
    for orig, nuevo in [('spro_fami_tieneinternet', 'internet'), ('spro_fami_tienecomputador', 'computador'),
                        ('spro_fami_tieneautomovil', 'automovil'), ('spro_fami_tienelavadora', 'lavadora'),
                        ('spro_fami_tienemotocicleta', 'motocicleta'), ('spro_fami_tieneserviciotv', 'tv_cerrada')]:
        df[nuevo] = (df[orig] == 'Si').astype(int) if orig in df.columns else np.nan
    df['horas_trabajo'] = df['spro_estu_horassemanatrabaja'].map(HORAS_MAP) \
        if 'spro_estu_horassemanatrabaja' in df.columns else np.nan
    df['cohorte_spro'] = df['periodo_sbpro'].astype(str).str[:4].astype(int)

    if 'spro_estu_nucleo_pregrado' in df.columns:
        nbc = df['spro_estu_nucleo_pregrado'].astype(str).str.strip().str.upper()
        orden = df.groupby(nbc)['s11_global'].mean().rank(method='first')
        df['nbc_cod'] = nbc.map(orden).astype(float)
    else:
        df['nbc_cod'] = np.nan
    return df, pd.DataFrame(atr)

ESPECIFICACIONES = {
    'pre_ingreso': ['s11_global', 's11_lectura', 's11_matematicas', 's11_cnaturales', 's11_sociales',
                    's11_ingles', 'inse_s11', 'genero', 'edu_madre', 'edu_padre', 'estrato',
                    'cohorte_spro', 'nbc_cod'],
    'ml19': ['s11_global', 's11_lectura', 's11_matematicas', 's11_cnaturales', 's11_sociales',
             's11_ingles', 'inse', 'genero', 'estrato', 'edu_madre', 'edu_padre', 'internet',
             'computador', 'automovil', 'lavadora', 'motocicleta', 'tv_cerrada', 'horas_trabajo',
             'cohorte_spro'],
}

# %% DML
def ols_cluster(y, X, groups):
    return sm.OLS(np.asarray(y, float), np.asarray(X, float)).fit(
        cov_type='cluster', cov_kwds={'groups': np.asarray(groups)})


def estimar_dml(df, y_col, x_cols, etiqueta_res, espec):
    d = df.dropna(subset=[y_col]).copy()
    datos = dml.DoubleMLData(d[[y_col, 'D', 'cluster'] + x_cols], y_col=y_col, d_cols='D',
                             x_cols=x_cols, cluster_cols='cluster', force_all_x_finite='allow-nan')
    filas, diag = [], {}
    n_ies = d['cluster'].nunique()

    def fila(modelo, est, t0):
        ci = est.confint().iloc[0]
        return {'resultado': etiqueta_res, 'especificacion': espec, 'estimador': modelo,
                'theta': float(est.coef[0]), 'se_cluster': float(est.se[0]),
                'ic95_inf': float(ci.iloc[0]), 'ic95_sup': float(ci.iloc[1]),
                'p_valor': float(est.pval[0]), 'n_estudiantes': len(d), 'n_ies': n_ies,
                'n_ies_oficiales': d.loc[d['D'] == 1, 'cluster'].nunique(),
                'segundos': round(time.time() - t0, 1)}

    np.random.seed(RANDOM_STATE)
    t0 = time.time()
    plr = dml.DoubleMLPLR(datos, ml_l=reg(), ml_m=clf(), n_folds=N_FOLDS, n_rep=N_REP)
    plr.fit(store_predictions=True)
    filas.append(fila('PLR (θ ponderado)', plr, t0))

    np.random.seed(RANDOM_STATE)
    t0 = time.time()
    irm = dml.DoubleMLIRM(datos, ml_g=reg(), ml_m=clf(), n_folds=N_FOLDS, n_rep=N_REP,
                          score='ATE', trimming_threshold=TRIMMING)
    irm.fit(store_predictions=True)
    filas.append(fila('IRM-ATE', irm, t0))

    np.random.seed(RANDOM_STATE)
    t0 = time.time()
    att = dml.DoubleMLIRM(datos, ml_g=reg(), ml_m=clf(), n_folds=N_FOLDS, n_rep=N_REP,
                          score='ATTE', trimming_threshold=TRIMMING)
    att.fit()
    filas.append(fila('IRM-ATT (oficiales)', att, t0))

    g0 = irm.predictions['ml_g0'][:, 0, 0]
    g1 = irm.predictions['ml_g1'][:, 0, 0]
    m = irm.predictions['ml_m'][:, 0, 0]
    yv, dv = d[y_col].values, d['D'].values
    diag = {'resultado': etiqueta_res, 'especificacion': espec,
            'rmse_g0_en_privadas': float(np.sqrt(mean_squared_error(yv[dv == 0], g0[dv == 0]))),
            'rmse_g1_en_oficiales': float(np.sqrt(mean_squared_error(yv[dv == 1], g1[dv == 1]))),
            'auc_propension': float(roc_auc_score(dv, m)),
            'pct_propension_truncada': float(np.mean((m < TRIMMING) | (m > 1 - TRIMMING)) * 100),
            'propension_p01': float(np.quantile(m, 0.01)), 'propension_p99': float(np.quantile(m, 0.99))}

    try:
        irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
        sp = irm.sensitivity_params
        diag['sens_rv_pct'] = float(np.ravel(sp['rv'])[0]) * 100
        diag['sens_rva_pct'] = float(np.ravel(sp['rva'])[0]) * 100
        diag['sens_theta_inf'] = float(np.ravel(sp['theta']['lower'])[0])
        diag['sens_theta_sup'] = float(np.ravel(sp['theta']['upper'])[0])
    except Exception as e:
        diag['sens_nota'] = f'no disponible: {type(e).__name__}: {e}'[:200]

    mc = np.clip(m, TRIMMING, 1 - TRIMMING)
    gamma = g1 - g0 + dv * (yv - g1) / mc - (1 - dv) * (yv - g0) / (1 - mc)
    pseudo = pd.DataFrame({'gamma': gamma, 'propension': m}, index=d.index)
    return filas, diag, pseudo

# %% Causal Forest
def causal_forest(df, y_col, x_mod, w_cols):
    cols = list(dict.fromkeys(x_mod + w_cols))
    d = df.dropna(subset=[y_col] + cols).copy()
    n_completos = len(d)
    if len(d) > CF_N_MAX:
        d = d.sample(CF_N_MAX, random_state=RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    ies = d['cluster'].unique()
    rng.shuffle(ies)
    mitad = set(ies[: len(ies) // 2])
    d['mitad'] = d['cluster'].map(lambda c: 0 if c in mitad else 1)

    d['cate'] = np.nan
    d['cate_inf'] = np.nan
    d['cate_sup'] = np.nan
    importancias = []
    for h in (0, 1):
        ent = d[d['mitad'] == h]
        pre = d[d['mitad'] != h]
        est = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                              cv=N_FOLDS, n_estimators=CF_N_ESTIMATORS, min_samples_leaf=CF_MIN_LEAF,
                              max_samples=0.45, honest=True, inference=True,
                              random_state=RANDOM_STATE, n_jobs=-1)
        est.fit(ent[y_col].values, ent['D'].values, X=ent[x_mod].values,
                W=ent[w_cols].values if w_cols else None, groups=ent['cluster'].values)
        d.loc[pre.index, 'cate'] = est.effect(pre[x_mod].values)
        lo, hi = est.effect_interval(pre[x_mod].values, alpha=0.05)
        d.loc[pre.index, 'cate_inf'] = lo
        d.loc[pre.index, 'cate_sup'] = hi
        importancias.append(est.feature_importances_)
    imp = pd.DataFrame({'variable': x_mod, 'importancia': np.mean(importancias, axis=0)}) \
        .sort_values('importancia', ascending=False)
    return d, imp, n_completos

# %% Heterogeneidad
def gate(gamma, grupos, clusters, nombre):
    ok = grupos.notna().values
    if isinstance(grupos.dtype, pd.CategoricalDtype):
        niveles = [str(c) for c in grupos.cat.categories if (grupos[ok] == c).any()]
    else:
        niveles = sorted(grupos[ok].astype(str).unique())
    g = grupos[ok].astype(str)
    X = np.column_stack([(g == lv).astype(float).values for lv in niveles])
    res = ols_cluster(gamma[ok], X, clusters[ok])
    ci = res.conf_int()
    filas = [{'subgrupo': nombre, 'nivel': lv, 'gate': res.params[i], 'se_cluster': res.bse[i],
              'ic95_inf': ci[i, 0], 'ic95_sup': ci[i, 1], 'n': int(X[:, i].sum())}
             for i, lv in enumerate(niveles)]
    p_igualdad = np.nan
    if len(niveles) > 1:
        R = np.zeros((len(niveles) - 1, len(niveles)))
        for i in range(len(niveles) - 1):
            R[i, 0], R[i, i + 1] = -1, 1
        p_igualdad = float(res.f_test(R).pvalue)
    for f in filas:
        f['p_igualdad_entre_niveles'] = p_igualdad
    return filas


def blp_y_sorted_gates(dcf, gamma_cf):
    tau = dcf['cate'].values
    cl = dcf['cluster'].values
    X = np.column_stack([np.ones_like(tau), tau - tau.mean()])
    r = ols_cluster(gamma_cf, X, cl)
    blp = {'beta1_ate': r.params[0], 'se_beta1': r.bse[0],
           'beta2_heterogeneidad': r.params[1], 'se_beta2': r.bse[1],
           'p_beta2_igual_0': float(r.pvalues[1]),
           'p_beta2_unilateral_mayor_0': float(r.pvalues[1] / 2 if r.params[1] > 0 else 1 - r.pvalues[1] / 2)}
    q = pd.qcut(pd.Series(tau).rank(method='first'), 4, labels=['Q1 (menor CATE)', 'Q2', 'Q3', 'Q4 (mayor CATE)'])
    Xq = np.column_stack([(q == lv).astype(float).values for lv in q.cat.categories])
    rq = ols_cluster(gamma_cf, Xq, cl)
    ci = rq.conf_int()
    filas = [{'cuartil': lv, 'gate': rq.params[i], 'se_cluster': rq.bse[i], 'ic95_inf': ci[i, 0],
              'ic95_sup': ci[i, 1], 'cate_medio_predicho': float(tau[(q == lv).values].mean()),
              'n': int(Xq[:, i].sum())} for i, lv in enumerate(q.cat.categories)]
    Rd = np.array([[-1, 0, 0, 1]])
    t = rq.t_test(Rd)
    blp['q4_menos_q1'] = float(np.ravel(t.effect)[0])
    blp['se_q4_menos_q1'] = float(np.ravel(t.sd)[0])
    blp['p_q4_menos_q1'] = float(np.ravel(t.pvalue)[0])
    return blp, pd.DataFrame(filas)

# %% Figuras
def figuras(res_ate, pseudo_p, df_p, gates, sorted_g, dcf):
    fig, ax = plt.subplots(figsize=(7, 4))
    m = pseudo_p['propension'].values
    dv = df_p.loc[pseudo_p.index, 'D'].values
    bins = np.linspace(0, 1, 51)
    ax.hist(m[dv == 0], bins=bins, alpha=0.6, density=True, label='Privada (D = 0)', color='#378ADD')
    ax.hist(m[dv == 1], bins=bins, alpha=0.6, density=True, label='Oficial (D = 1)', color='#D85A30')
    ax.axvline(TRIMMING, color='k', lw=0.6, ls='--'); ax.axvline(1 - TRIMMING, color='k', lw=0.6, ls='--')
    ax.set_xlabel('Propensión estimada P(IES oficial | X): fuera de pliegue')
    ax.set_ylabel('Densidad'); ax.set_title('Soporte común (especificación principal, resultado global)')
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{RUTA_SALIDA}/fig_c3_1_soporte_comun.png', dpi=300); plt.close()

    r = res_ate.copy()
    r['etiqueta'] = r['resultado'] + ' · ' + r['estimador'] + ' · ' + r['especificacion']
    r = r.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(r) + 1.2))
    col = np.where(r['especificacion'] == ESPEC_PRINCIPAL, '#1D2B53', '#9AA3B5')
    ax.errorbar(r['theta'], range(len(r)), xerr=[r['theta'] - r['ic95_inf'], r['ic95_sup'] - r['theta']],
                fmt='none', ecolor=col, elinewidth=1.2)
    ax.scatter(r['theta'], range(len(r)), c=col, s=18, zorder=3)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r['etiqueta'], fontsize=7)
    ax.set_xlabel('Efecto de asistir a una IES oficial (puntos de Saber Pro), IC 95 % agrupado por IES')
    ax.set_title('Efecto promedio del sector institucional (DML)')
    plt.tight_layout(); plt.savefig(f'{RUTA_SALIDA}/fig_c3_2_efecto_promedio.png', dpi=300); plt.close()

    subg = gates['subgrupo'].unique()
    fig, axes = plt.subplots(1, len(subg), figsize=(3.2 * len(subg), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subg):
        g = gates[gates['subgrupo'] == s]
        x = np.arange(len(g))
        ax.errorbar(x, g['gate'], yerr=[g['gate'] - g['ic95_inf'], g['ic95_sup'] - g['gate']],
                    fmt='o', color='#1D2B53', capsize=3)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(g['nivel'], rotation=30, ha='right', fontsize=7)
        ax.set_title(f"{s}\np(igualdad) = {g['p_igualdad_entre_niveles'].iloc[0]:.3f}", fontsize=9)
    axes[0].set_ylabel('GATE (puntos), IC 95 % agrupado')
    plt.tight_layout(); plt.savefig(f'{RUTA_SALIDA}/fig_c3_3_gate_subgrupos.png', dpi=300); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(sorted_g))
    axes[0].errorbar(x, sorted_g['gate'], yerr=[sorted_g['gate'] - sorted_g['ic95_inf'],
                                                sorted_g['ic95_sup'] - sorted_g['gate']],
                     fmt='o', color='#1D2B53', capsize=3, label='GATE (AIPW)')
    axes[0].scatter(x, sorted_g['cate_medio_predicho'], marker='x', color='#D85A30', label='CATE medio predicho')
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(sorted_g['cuartil'], fontsize=8)
    axes[0].set_ylabel('Puntos de Saber Pro'); axes[0].set_title('GATES ordenados por cuartil del CATE')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(dcf['cate'], bins=60, color='#378ADD', alpha=0.8)
    axes[1].axvline(0, color='k', lw=0.6)
    axes[1].set_xlabel('CATE predicho fuera de muestra (puntos)'); axes[1].set_title('Distribución del CATE')
    plt.tight_layout(); plt.savefig(f'{RUTA_SALIDA}/fig_c3_4_sorted_gates.png', dpi=300); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, v, lab in [(axes[0], 'inse_s11', 'INSE (Saber 11)'), (axes[1], 's11_global', 'Puntaje global Saber 11')]:
        dec = pd.qcut(dcf[v].rank(method='first'), 10, labels=False)
        agg = dcf.groupby(dec).agg(x=(v, 'mean'), cate=('cate', 'mean'),
                                   lo=('cate_inf', 'mean'), hi=('cate_sup', 'mean'))
        ax.plot(agg['x'], agg['cate'], 'o-', color='#1D2B53')
        ax.fill_between(agg['x'], agg['lo'], agg['hi'], color='#1D2B53', alpha=0.12,
                        label='IC del bosque (no agrupado)')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel(f'{lab} (media por decil)'); ax.set_ylabel('CATE medio (puntos)')
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Heterogeneidad del efecto del sector según las condiciones de entrada', fontsize=11)
    plt.tight_layout(); plt.savefig(f'{RUTA_SALIDA}/fig_c3_5_cate_vs_entrada.png', dpi=300); plt.close()

# %% Resumen ejecutivo
def resumen(atr, res_ate, diag, gates, blp, sorted_g, imp, n_cf, n_cf_completos, cobertura_inse):
    L = []
    p = lambda s='': L.append(s)
    p('RESUMEN EJECUTIVO: CAPA 3 (DML + CAUSAL FORESTS)')
    fin = atr.iloc[-1]
    p(f"\n1. MUESTRA: {fin['estudiantes']:,} estudiantes de {fin['IES']} IES.")
    p(f"   Cobertura del INSE de Saber 11 en la muestra: {cobertura_inse:.1%} "
      f"(los faltantes se tratan de forma nativa en LightGBM; el bosque usa casos completos).")
    princ = res_ate[(res_ate['especificacion'] == ESPEC_PRINCIPAL)]
    r0 = princ[(princ['resultado'] == RESULTADO_CF) & (princ['estimador'] == 'IRM-ATE')].iloc[0]
    p(f"   IES oficiales: {r0['n_ies_oficiales']} de {r0['n_ies']} (conglomerados efectivos para la inferencia).")

    p('\n2. EFECTO PROMEDIO DE ASISTIR A UNA IES OFICIAL (puntos de Saber Pro, IC 95 % agrupado por IES):')
    for _, r in res_ate.iterrows():
        p(f"   [{r['especificacion']}] {r['resultado']:>20s} · {r['estimador']:<20s} "
          f"θ = {r['theta']:+.2f} (EE {r['se_cluster']:.2f}; IC {r['ic95_inf']:+.2f} a {r['ic95_sup']:+.2f}; p = {r['p_valor']:.3f})")

    p('\n3. DIAGNÓSTICOS (especificación principal):')
    for dg in diag:
        if dg['especificacion'] != ESPEC_PRINCIPAL:
            continue
        p(f"   {dg['resultado']}: AUC propensión = {dg['auc_propension']:.3f}; "
          f"propensión p1-p99 = [{dg['propension_p01']:.3f}, {dg['propension_p99']:.3f}]; "
          f"truncada = {dg['pct_propension_truncada']:.2f}%")
        if 'sens_rv_pct' in dg:
            p(f"      Sensibilidad: valor de robustez RV = {dg['sens_rv_pct']:.1f}% (RVa = {dg['sens_rva_pct']:.1f}%); "
              f"con confusión de 3%/3%, θ entre {dg['sens_theta_inf']:+.2f} y {dg['sens_theta_sup']:+.2f}")
        elif 'sens_nota' in dg:
            p(f"      Sensibilidad {dg['sens_nota']}")

    p(f"\n4. HETEROGENEIDAD (resultado: {RESULTADO_CF}; bosque sobre {n_cf:,} estudiantes de {n_cf_completos:,} con casos completos):")
    p(f"   Prueba BLP: β2 = {blp['beta2_heterogeneidad']:.3f} (EE {blp['se_beta2']:.3f}); "
      f"p(β2 = 0) = {blp['p_beta2_igual_0']:.4f}; p unilateral (β2 > 0) = {blp['p_beta2_unilateral_mayor_0']:.4f}")
    p(f"   GATES ordenados: Q4 − Q1 = {blp['q4_menos_q1']:+.2f} puntos (EE {blp['se_q4_menos_q1']:.2f}; p = {blp['p_q4_menos_q1']:.4f})")
    for _, r in sorted_g.iterrows():
        p(f"      {r['cuartil']:<16s} GATE = {r['gate']:+.2f} [{r['ic95_inf']:+.2f}, {r['ic95_sup']:+.2f}]  "
          f"(CATE medio predicho {r['cate_medio_predicho']:+.2f})")
    p('   GATE por subgrupos definidos a priori:')
    for s in gates['subgrupo'].unique():
        g = gates[gates['subgrupo'] == s]
        txt = '; '.join(f"{a}: {b:+.2f}" for a, b in zip(g['nivel'], g['gate']))
        p(f"      {s}: {txt}  (p igualdad = {g['p_igualdad_entre_niveles'].iloc[0]:.4f})")
    p('   Importancia de los moduladores en el bosque: ' +
      ', '.join(f"{a} ({b:.2f})" for a, b in zip(imp['variable'], imp['importancia'])))

    texto = '\n'.join(L)
    print(texto)
    with open(f'{RUTA_SALIDA}/resumen_ejecutivo.txt', 'w', encoding='utf-8') as f:
        f.write(texto)

# %% Ejecución
def ejecutar():
    t_inicio = time.time()
    df_raw = pd.read_csv(RUTA_TABLA, low_memory=False)
    print(f'Tabla cargada: {len(df_raw):,} filas')
    df, atr = preparar(df_raw)
    atr.to_csv(f'{RUTA_SALIDA}/tabla_atricion_capa3.csv', index=False)
    print(atr.to_string(index=False))
    cobertura_inse = float(df['inse_s11'].notna().mean())
    if cobertura_inse < 0.5:
        print(f'Cobertura del INSE de Saber 11: {cobertura_inse:.1%}')

    resultados = {k: v for k, v in RESULTADOS.items() if v in df.columns}
    especs = [ESPEC_PRINCIPAL] + (['ml19'] if CORRER_ROBUSTEZ_ML19 else [])

    filas, diag, pseudo_p = [], [], None
    for espec in especs:
        xcols = [c for c in ESPECIFICACIONES[espec] if df[c].notna().any()]
        for etq, ycol in resultados.items():
            if espec != ESPEC_PRINCIPAL and etq != RESULTADO_CF:
                continue
            print(f'\n[DML] {espec} · {etq} ...')
            f, dg, ps = estimar_dml(df, ycol, xcols, etq, espec)
            filas += f
            diag.append(dg)
            for x in f:
                print(f"   {x['estimador']:<22s} θ = {x['theta']:+.3f} (EE {x['se_cluster']:.3f})  [{x['segundos']} s]")
            if espec == ESPEC_PRINCIPAL and etq == RESULTADO_CF:
                pseudo_p = ps
    res_ate = pd.DataFrame(filas)
    res_ate.to_csv(f'{RUTA_SALIDA}/tabla_efecto_promedio.csv', index=False)
    pd.DataFrame(diag).to_csv(f'{RUTA_SALIDA}/tabla_diagnosticos_nuisance_sensibilidad.csv', index=False)

    dp = df.loc[pseudo_p.index]
    clusters = dp['cluster'].values
    gamma = pseudo_p['gamma'].values
    subgrupos = {
        'INSE (Saber 11)': pd.qcut(dp['inse_s11'], 3, labels=['Bajo', 'Medio', 'Alto']),
        'Saber 11 global': pd.qcut(dp['s11_global'], 3, labels=['Bajo', 'Medio', 'Alto']),
        'Estrato': pd.cut(dp['estrato'], [0, 2, 4, 6], labels=['1-2', '3-4', '5-6']),
        'Género': dp['genero'].map({1: 'Hombre', 0: 'Mujer'}),
    }
    gates = []
    for nombre, grp in subgrupos.items():
        gates += gate(gamma, pd.Series(grp, index=dp.index), clusters, nombre)
    gates = pd.DataFrame(gates)
    gates.to_csv(f'{RUTA_SALIDA}/tabla_gate_subgrupos.csv', index=False)

    print('\n[CF] Causal Forest con predicción fuera de muestra por mitades de IES...')
    t0 = time.time()
    x_mod = [c for c in CF_MODULADORES if df[c].notna().any()]
    w_cols = [c for c in ESPECIFICACIONES[ESPEC_PRINCIPAL] if c not in x_mod and df[c].notna().any()]
    dcf, imp, n_cf_completos = causal_forest(df, RESULTADOS[RESULTADO_CF], x_mod, w_cols)
    print(f'   listo en {time.time() - t0:.0f} s sobre {len(dcf):,} estudiantes')
    imp.to_csv(f'{RUTA_SALIDA}/tabla_importancia_moduladores_cf.csv', index=False)
    dcf_out = dcf[['ies_id', 'D'] + x_mod + ['cate', 'cate_inf', 'cate_sup']]
    dcf_out.to_csv(f'{RUTA_SALIDA}/cate_por_estudiante_muestra_cf.csv', index=False)
    dcf_out.groupby('ies_id').agg(D=('D', 'first'), n=('cate', 'size'), cate_medio=('cate', 'mean'),
                                  cate_de=('cate', 'std')).reset_index() \
        .to_csv(f'{RUTA_SALIDA}/cate_resumen_por_ies.csv', index=False)

    gamma_cf = pseudo_p.loc[dcf.index, 'gamma'].values
    blp, sorted_g = blp_y_sorted_gates(dcf, gamma_cf)
    pd.DataFrame([blp]).to_csv(f'{RUTA_SALIDA}/tabla_blp_calibracion.csv', index=False)
    sorted_g.to_csv(f'{RUTA_SALIDA}/tabla_sorted_gates.csv', index=False)

    figuras(res_ate, pseudo_p, df, gates, sorted_g, dcf)
    resumen(atr, res_ate, diag, gates, blp, sorted_g, imp, len(dcf), n_cf_completos, cobertura_inse)

    with open(f'{RUTA_SALIDA}/parametros_capa3.json', 'w', encoding='utf-8') as f:
        json.dump({'n_folds': N_FOLDS, 'n_rep': N_REP, 'trimming': TRIMMING,
                   'resultados': resultados, 'especificaciones': {e: ESPECIFICACIONES[e] for e in especs},
                   'cf_moduladores': x_mod, 'cf_controles_W': w_cols, 'cf_n_max': CF_N_MAX,
                   'cf_n_estimators': CF_N_ESTIMATORS, 'cf_min_leaf': CF_MIN_LEAF, 'lgbm': LGBM_PARAMS,
                   'doubleml': dml.__version__, 'tiempo_total_min': round((time.time() - t_inicio) / 60, 1)},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nArchivos en {RUTA_SALIDA}/, tiempo total {(time.time() - t_inicio) / 60:.1f} min')


if __name__ == '__main__':
    ejecutar()
