# %% Réplica de la medición del ICFES
import os
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'sans-serif',
})

# %% Configuración
RUTA_TABLA = r"C:\Users\usuario\Desktop\DataICFES\datos\output\tabla_analitica_VAE.csv"

RUTA_SABERPRO_CRUDO = None
RUTA_SALIDA = "salida_oe5"

APLICAR_FILTRO_AGREGADO = False
os.makedirs(RUTA_SALIDA, exist_ok=True)

RANDOM_STATE = 42

PARAMS = {
    'min_estudiantes_inbc': 25,
    'min_pct_cruce': 0.40,
    'max_modulos_cero': 2,
    'min_ies_por_nbc': 5,
    'max_vecinos': 15,
    'ventana_min': 4,
    'ventana_max': 8,
    'ventana_medicina': 9,
    'exigir_ambos_anios': True,
}

COMPETENCIAS = {
    'lectura_critica': 'spro_mod_lectura_critica_punt',
    'razona_cuantitativo': 'spro_mod_razona_cuantitat_punt',
    'ingles': 'spro_mod_ingles_punt',
}

REGRESORES_S11 = [
    's11_punt_matematicas',
    's11_punt_lectura_critica',
    's11_punt_c_naturales',
    's11_punt_sociales_ciudadanas',
    's11_punt_ingles',
]

MODULOS_SPRO = [
    'spro_mod_lectura_critica_punt',
    'spro_mod_razona_cuantitat_punt',
    'spro_mod_competen_ciudada_punt',
    'spro_mod_comuni_escrita_punt',
    'spro_mod_ingles_punt',
]

ICC_REFERENCIA_ICFES_2017_2018 = {
    'ADMINISTRACION': (0.1818, 0.1630, 0.3470, 138),
    'ANTROPOLOGIA, ARTES LIBERALES': (0.1821, 0.1977, 0.3724, 9),
    'ARQUITECTURA': (0.1927, 0.2094, 0.2988, 48),
    'ARTES PLASTICAS, VISUALES Y AFINES': (0.1350, 0.1334, 0.2796, 23),
    'BACTERIOLOGIA': (0.2244, 0.2289, 0.2189, 14),
    'BIOLOGIA, MICROBIOLOGIA Y AFINES': (0.2570, 0.2351, 0.3668, 32),
    'CIENCIA POLITICA, RELACIONES INTERNACIONALES': (0.1136, 0.1042, 0.3340, 24),
    'COMUNICACION SOCIAL, PERIODISMO Y AFINES': (0.1187, 0.1254, 0.2400, 52),
    'CONTADURIA PUBLICA': (0.1682, 0.1554, 0.2342, 102),
    'DEPORTES, EDUCACION FISICA Y RECREACION': (0.1557, 0.1517, 0.1616, 14),
    'DERECHO Y AFINES': (0.1802, 0.2239, 0.3317, 93),
    'DISENO': (0.1771, 0.1494, 0.2466, 45),
    'ECONOMIA': (0.2680, 0.2126, 0.3427, 68),
    'EDUCACION': (0.1596, 0.2007, 0.2786, 72),
    'ENFERMERIA': (0.2473, 0.2416, 0.2170, 54),
    'FILOSOFIA, TEOLOGIA Y AFINES': (0.2376, 0.3371, 0.2840, 6),
    'FISICA': (0.1222, 0.2782, 0.3623, 5),
    'GEOGRAFIA, HISTORIA': (0.2428, 0.3590, 0.4390, 13),
    'INGENIERIA ADMINISTRATIVA Y AFINES': (0.3385, 0.2280, 0.3109, 10),
    'INGENIERIA AGROINDUSTRIAL, ALIMENTOS Y AFINES': (0.1149, 0.1127, 0.1628, 22),
    'INGENIERIA AGRONOMICA, PECUARIA Y AFINES': (0.3418, 0.2832, 0.3002, 16),
    'INGENIERIA AGRICOLA, FORESTAL Y AFINES': (0.3331, 0.2440, 0.3082, 13),
    'INGENIERIA AMBIENTAL, SANITARIA Y AFINES': (0.2406, 0.2160, 0.3001, 54),
    'INGENIERIA BIOMEDICA Y AFINES': (0.1720, 0.2021, 0.3180, 11),
    'INGENIERIA CIVIL Y AFINES': (0.2666, 0.2267, 0.3426, 57),
    'INGENIERIA DE MINAS, METALURGIA Y AFINES': (0.2383, 0.0856, 0.3397, 9),
    'INGENIERIA DE SISTEMAS, TELEMATICA Y AFINES': (0.2893, 0.2424, 0.4098, 80),
    'INGENIERIA ELECTRONICA, TELECOMUNICACIONES Y AFINES': (0.2383, 0.2084, 0.2839, 51),
    'INGENIERIA ELECTRICA Y AFINES': (0.2158, 0.2065, 0.2585, 20),
    'INGENIERIA INDUSTRIAL Y AFINES': (0.2410, 0.1700, 0.3266, 83),
    'INGENIERIA MECANICA Y AFINES': (0.2626, 0.2049, 0.3554, 39),
    'INGENIERIA QUIMICA Y AFINES': (0.1798, 0.1767, 0.2664, 17),
    'INSTRUMENTACION QUIRURGICA': (0.1936, 0.1747, 0.2100, 14),
    'LENGUAS MODERNAS, LITERATURA, LINGUISTICA Y AFINES': (0.1405, 0.1551, 0.3311, 10),
    'MATEMATICAS, ESTADISTICA Y AFINES': (0.2029, 0.2767, 0.4407, 8),
    'MEDICINA': (0.2210, 0.2266, 0.3316, 52),
    'MEDICINA VETERINARIA': (0.1884, 0.2321, 0.2821, 25),
    'MUSICA': (0.0508, 0.0823, 0.3011, 16),
    'NUTRICION Y DIETETICA': (0.3071, 0.2411, 0.2413, 9),
    'ODONTOLOGIA': (0.1872, 0.1864, 0.2060, 21),
    'OPTOMETRIA, OTROS PROGRAMAS DE CIENCIAS DE LA SALUD': (0.0680, 0.0511, 0.0612, 7),
    'OTRAS INGENIERIAS': (0.3169, 0.3275, 0.4611, 16),
    'PSICOLOGIA': (0.2211, 0.2204, 0.3566, 80),
    'PUBLICIDAD Y AFINES': (0.0676, 0.0550, 0.1222, 13),
    'QUIMICA Y AFINES': (0.1648, 0.1999, 0.2729, 15),
    'SALUD PUBLICA': (0.0141, 0.0118, 0.0204, 4),
    'SOCIOLOGIA, TRABAJO SOCIAL Y AFINES': (0.2851, 0.3613, 0.4318, 39),
    'TERAPIAS': (0.1988, 0.2107, 0.2343, 29),
    'ZOOTECNIA': (0.2177, 0.1881, 0.2794, 14),
}


def normalizar_nbc(s):
    if not isinstance(s, str):
        return None
    txt = s.upper().strip()
    for a, b in zip('ÁÉÍÓÚÜÑ', 'AEIOUUN'):
        txt = txt.replace(a, b)
    return ' '.join(txt.split())

# %% Carga y preparación
def cargar_y_preparar(ruta=RUTA_TABLA):
    df = pd.read_csv(ruta, low_memory=False)
    print(f"Tabla cargada: {len(df):,} filas x {len(df.columns)} columnas")
    if APLICAR_FILTRO_AGREGADO and 'spro_estu_agregado' in df.columns:
        df = df[df['spro_estu_agregado'] == 'S'].copy()
        print(f"Con marca de agregado: {len(df):,} filas")

    num_cols = REGRESORES_S11 + MODULOS_SPRO + [
        's11_punt_global', 'spro_punt_global',
        's11_estu_inse_individual', 'spro_estu_inse_individual',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['ies_id'] = df['spro_inst_cod_institucion'].astype(str)
    df['nbc'] = df['spro_estu_nucleo_pregrado'].astype(str).str.strip()
    df['nbc_norm'] = df['nbc'].map(normalizar_nbc)
    df['inbc_id'] = df['ies_id'] + '__' + df['nbc_norm'].fillna('SIN_NBC')

    df['anio_spro'] = df['periodo_sbpro'].astype(str).str[:4].astype(int)
    df['anio_s11'] = df['periodo_sb11'].astype(str).str[:4].astype(int)
    df['anios_entre_pruebas'] = df['anio_spro'] - df['anio_s11']

    mods = [m for m in MODULOS_SPRO if m in df.columns]
    df['n_modulos_cero'] = sum(
        ((df[m].isna()) | (df[m] == 0)).astype(int) for m in mods
    )

    print(f"  IES: {df['ies_id'].nunique()} | NBC: {df['nbc_norm'].nunique()} "
          f"| INBC: {df['inbc_id'].nunique()}")
    print(f"  Años Saber Pro presentes: {sorted(df['anio_spro'].unique())}")
    return df

# %% Construcción de períodos bianuales
def construir_periodos(df):
    anios = sorted(df['anio_spro'].unique())
    periodos = [(a, a + 1) for a in anios if (a + 1) in anios]

    print("Períodos bianuales construibles:")
    if not periodos:
        print("  NINGUNO. No hay años consecutivos en la tabla analítica.")
    for a1, a2 in periodos:
        n = df[df['anio_spro'].isin([a1, a2])].shape[0]
        print(f"  {a1}-{a2}: {n:,} estudiantes")

    huerfanos = [a for a in anios
                 if not any(a in p for p in periodos)]
    if huerfanos:
        print(f"  Años sin pareja (excluidos): {huerfanos}")
    return periodos

# %% Filtros del ICFES
def cargar_denominadores_spro(ruta_crudo, anios):
    if ruta_crudo is None or not os.path.isdir(ruta_crudo):
        return None

    piezas = []
    for f in sorted(os.listdir(ruta_crudo)):
        if not (f.endswith('.txt') or f.endswith('.csv')):
            continue
        cols = ['periodo', 'inst_cod_institucion', 'estu_nucleo_pregrado']
        try:
            d = pd.read_csv(os.path.join(ruta_crudo, f), sep=';',
                            usecols=lambda c: c in cols, low_memory=False)
        except Exception as e:
            print(f"  No se pudo leer {f}: {e}")
            continue
        if not set(cols).issubset(d.columns):
            continue
        d['anio_spro'] = d['periodo'].astype(str).str[:4].astype(int)
        d = d[d['anio_spro'].isin(anios)]
        if len(d):
            piezas.append(d)

    if not piezas:
        return None

    crudo = pd.concat(piezas, ignore_index=True)
    crudo['inbc_id'] = (crudo['inst_cod_institucion'].astype(str) + '__' +
                        crudo['estu_nucleo_pregrado'].astype(str).map(normalizar_nbc).fillna('SIN_NBC'))
    denom = (crudo.groupby(['inbc_id', 'anio_spro'])
                  .size().reset_index(name='n_total_spro'))
    print(f"  Denominadores cargados para {denom['inbc_id'].nunique():,} INBC")
    return denom

def aplicar_filtros_icfes(df, periodo, params, denominadores=None, verbose=True):
    a1, a2 = periodo
    reg = []

    def registrar(etapa, d, nota=''):
        reg.append({
            'etapa': etapa,
            'estudiantes': len(d),
            'INBC': d['inbc_id'].nunique() if len(d) else 0,
            'IES': d['ies_id'].nunique() if len(d) else 0,
            'NBC': d['nbc_norm'].nunique() if len(d) else 0,
            'nota': nota,
        })

    d = df[df['anio_spro'].isin([a1, a2])].copy()
    registrar('0. Período bianual', d)

    es_medicina = d['nbc_norm'].str.contains('MEDICINA', na=False) & \
                  ~d['nbc_norm'].str.contains('VETERINARIA', na=False)
    tope = np.where(es_medicina, params['ventana_medicina'], params['ventana_max'])
    d = d[(d['anios_entre_pruebas'] >= params['ventana_min']) &
          (d['anios_entre_pruebas'] <= tope)]
    registrar(f"a. Ventana {params['ventana_min']}-{params['ventana_max']} años "
              f"({params['ventana_medicina']} medicina)", d)

    d = d[d['n_modulos_cero'] < params['max_modulos_cero']]
    registrar(f"b. <{params['max_modulos_cero']} módulos en 0 o sin calificar", d)

    if params['exigir_ambos_anios']:
        conteo = d.groupby(['inbc_id', 'anio_spro']).size().unstack(fill_value=0)
        for a in (a1, a2):
            if a not in conteo.columns:
                conteo[a] = 0
        validos = conteo[(conteo[a1] >= params['min_estudiantes_inbc']) &
                         (conteo[a2] >= params['min_estudiantes_inbc'])].index
        nota = 'exigido en ambos años'
    else:
        conteo = d.groupby('inbc_id').size()
        validos = conteo[conteo >= params['min_estudiantes_inbc']].index
        nota = 'exigido sobre el período agregado'
    d = d[d['inbc_id'].isin(validos)]
    registrar(f"c. INBC con >={params['min_estudiantes_inbc']} estudiantes", d, nota)

    if denominadores is not None:
        cruzados = (d.groupby(['inbc_id', 'anio_spro']).size()
                     .reset_index(name='n_cruzados'))
        m = cruzados.merge(denominadores, on=['inbc_id', 'anio_spro'], how='left')
        m['pct'] = m['n_cruzados'] / m['n_total_spro']
        if params['exigir_ambos_anios']:
            ok = (m.groupby('inbc_id')['pct']
                    .agg(lambda s: s.notna().all() and (s >= params['min_pct_cruce']).all()))
        else:
            ok = m.groupby('inbc_id')['pct'].mean() >= params['min_pct_cruce']
        d = d[d['inbc_id'].isin(ok[ok].index)]
        registrar(f"d. Cruce >={params['min_pct_cruce']:.0%} de la población INBC", d)
    else:
        registrar('d. Cruce >=40% (omitido)', d, 'sin denominador')

    ies_por_nbc = d.groupby('nbc_norm')['ies_id'].nunique()
    nbc_ok = ies_por_nbc[ies_por_nbc >= params['min_ies_por_nbc']].index
    d = d[d['nbc_norm'].isin(nbc_ok)]
    registrar(f"e. NBC con >={params['min_ies_por_nbc']} instituciones", d)

    atricion = pd.DataFrame(reg)
    if verbose:
        print(f"\nTabla de atrición, período {a1}-{a2}")
        print(atricion.to_string(index=False))
    return d, atricion

# %% Modelo nulo por NBC
def estimar_modelo_nulo(d, competencias=COMPETENCIAS):
    filas = []
    nbcs = sorted(d['nbc_norm'].dropna().unique())
    for i, nbc in enumerate(nbcs, 1):
        sub_nbc = d[d['nbc_norm'] == nbc]
        for comp, ycol in competencias.items():
            sub = sub_nbc.dropna(subset=[ycol, 'inbc_id'])
            if sub['inbc_id'].nunique() < 2 or len(sub) < 30:
                continue
            try:
                res = smf.mixedlm(f"{ycol} ~ 1", data=sub,
                                  groups=sub['inbc_id']).fit(reml=True)
                tau = float(res.cov_re.iloc[0, 0])
                sigma2 = float(res.scale)
                filas.append({
                    'nbc_norm': nbc, 'competencia': comp,
                    'tau': tau, 'sigma2': sigma2,
                    'ICC': tau / (tau + sigma2),
                    'n_estudiantes': len(sub),
                    'n_inbc': sub['inbc_id'].nunique(),
                    'n_ies': sub['ies_id'].nunique(),
                })
            except Exception as e:
                print(f"    [nulo] fallo en {nbc} / {comp}: {e}")
        if i % 10 == 0:
            print(f"  modelo nulo: {i}/{len(nbcs)} NBC")
    return pd.DataFrame(filas)

# %% Modelo-espejo
def _var_posterior_blup(tau, sigma2, n_j):
    return (tau * sigma2) / (n_j * tau + sigma2)

def estimar_valor_agregado(d, competencias=COMPETENCIAS, regresores=REGRESORES_S11):
    coefs, blups = [], []
    nbcs = sorted(d['nbc_norm'].dropna().unique())

    for i, nbc in enumerate(nbcs, 1):
        sub_nbc = d[d['nbc_norm'] == nbc]
        for comp, ycol in competencias.items():
            sub = sub_nbc.dropna(subset=[ycol] + regresores + ['inbc_id'])
            if sub['inbc_id'].nunique() < 2 or len(sub) < 50:
                continue
            formula = f"{ycol} ~ " + " + ".join(regresores)
            try:
                res = smf.mixedlm(formula, data=sub,
                                  groups=sub['inbc_id']).fit(reml=True)
            except Exception as e:
                print(f"    [VA] fallo en {nbc} / {comp}: {e}")
                continue

            tau = float(res.cov_re.iloc[0, 0])
            sigma2 = float(res.scale)

            fila = {'nbc_norm': nbc, 'competencia': comp,
                    'n_estudiantes': len(sub), 'n_inbc': sub['inbc_id'].nunique(),
                    'tau': tau, 'sigma2': sigma2,
                    'varianza_colapsada': tau <= 1e-8}
            for v in ['Intercept'] + regresores:
                if v in res.fe_params.index:
                    fila[f'b_{v}'] = float(res.fe_params[v])
                    fila[f'se_{v}'] = float(res.bse_fe[v])
            coefs.append(fila)

            if tau <= 1e-8:
                print(f"    [VA] {nbc} / {comp}: varianza entre INBC ~ 0; "
                      f"sin BLUPs estimables (n={len(sub)}, "
                      f"INBC={sub['inbc_id'].nunique()})")
                continue

            try:
                re_dict = res.random_effects
            except Exception as e:
                print(f"    [VA] {nbc} / {comp}: no se pudieron extraer BLUPs: {e}")
                continue

            re_dict = res.random_effects
            n_por_inbc = sub.groupby('inbc_id').size()
            medias = sub.groupby('inbc_id').agg(
                s11_global_medio=('s11_punt_global', 'mean'),
                inse_s11_medio=('s11_estu_inse_individual', 'mean'),
                spro_bruto_medio=(ycol, 'mean'),
            )
            for inbc, ef in re_dict.items():
                n_j = int(n_por_inbc.get(inbc, 0))
                if n_j == 0:
                    continue
                va = float(ef.iloc[0])
                se = float(np.sqrt(_var_posterior_blup(tau, sigma2, n_j)))
                blups.append({
                    'inbc_id': inbc,
                    'ies_id': str(inbc).split('__')[0],
                    'nbc_norm': nbc,
                    'competencia': comp,
                    'VA_espejo': va,
                    'se_VA': se,
                    'ic_inf': va - 1.96 * se,
                    'ic_sup': va + 1.96 * se,
                    'n_estudiantes': n_j,
                    's11_global_medio': float(medias.loc[inbc, 's11_global_medio']),
                    'inse_s11_medio': float(medias.loc[inbc, 'inse_s11_medio'])
                        if pd.notna(medias.loc[inbc, 'inse_s11_medio']) else np.nan,
                    'spro_bruto_medio': float(medias.loc[inbc, 'spro_bruto_medio']),
                })
        if i % 10 == 0:
            print(f"  valor agregado: {i}/{len(nbcs)} NBC")

    return pd.DataFrame(coefs), pd.DataFrame(blups)

# %% Vecindades de comparación
def construir_vecindades(va_df, max_vecinos=None):
    max_vecinos = max_vecinos or PARAMS['max_vecinos']
    pares = []
    for (nbc, comp), g in va_df.groupby(['nbc_norm', 'competencia']):
        g = g.reset_index(drop=True)
        for _, ref in g.iterrows():
            otros = g[g['inbc_id'] != ref['inbc_id']].copy()
            if otros.empty:
                continue
            otros['dist'] = (otros['s11_global_medio'] - ref['s11_global_medio']).abs()
            vecinos = otros.nsmallest(max_vecinos, 'dist')
            for _, v in vecinos.iterrows():
                pares.append({
                    'nbc_norm': nbc, 'competencia': comp,
                    'inbc_referencia': ref['inbc_id'],
                    'inbc_vecino': v['inbc_id'],
                    'VA_referencia': ref['VA_espejo'],
                    'VA_vecino': v['VA_espejo'],
                    'dist_s11': v['dist'],
                })
    return pd.DataFrame(pares)

# %% Aporte Relativo
def calcular_aporte_relativo(vecindades, va_df):
    ar = vecindades.copy()
    ar['aporte_relativo'] = ar['VA_vecino'] - ar['VA_referencia']

    resumen = []
    for (nbc, comp, ref), g in ar.groupby(['nbc_norm', 'competencia', 'inbc_referencia']):
        valores = np.concatenate([[0.0], g['aporte_relativo'].values])
        pct = (valores < 0).sum() / max(len(valores) - 1, 1)
        resumen.append({
            'nbc_norm': nbc, 'competencia': comp, 'inbc_id': ref,
            'pct_vecindad': pct,
            'n_vecinos': len(g),
            'ar_medio_vecinos': g['aporte_relativo'].mean(),
            'amplitud_s11_vecindad': g['dist_s11'].max(),
        })
    resumen = pd.DataFrame(resumen)
    return ar, resumen.merge(
        va_df[['inbc_id', 'competencia', 'VA_espejo', 'se_VA', 'n_estudiantes',
               's11_global_medio', 'inse_s11_medio', 'spro_bruto_medio', 'ies_id']],
        on=['inbc_id', 'competencia'], how='left')

# %% Propuesta
def estimar_propuesta(d, competencias=COMPETENCIAS, regresores=REGRESORES_S11,
                      umbral_cobertura=0.50):
    inse_col = 's11_estu_inse_individual'
    cobertura = d[inse_col].notna().mean() if inse_col in d.columns else 0.0
    print(f"  Cobertura INSE Saber 11: {cobertura:.1%}")
    if cobertura < umbral_cobertura:
        alt = 'spro_estu_inse_individual'
        print(f"  Cobertura insuficiente, se usa {alt}.")
        inse_col = alt

    filas = []
    nbcs = sorted(d['nbc_norm'].dropna().unique())
    for i, nbc in enumerate(nbcs, 1):
        sub_nbc = d[d['nbc_norm'] == nbc]
        for comp, ycol in competencias.items():
            sub = sub_nbc.dropna(subset=[ycol] + regresores + [inse_col, 'inbc_id'])
            if sub['inbc_id'].nunique() < 2 or len(sub) < 50:
                continue
            formula = f"{ycol} ~ " + " + ".join(regresores) + f" + {inse_col}"
            try:
                res = smf.mixedlm(formula, data=sub,
                                  groups=sub['inbc_id']).fit(reml=True)
            except Exception as e:
                print(f"    [propuesta] fallo en {nbc} / {comp}: {e}")
                continue
            tau = float(res.cov_re.iloc[0, 0])
            sigma2 = float(res.scale)
            n_por_inbc = sub.groupby('inbc_id').size()
            if tau <= 1e-8:
                continue
            try:
                re_items = res.random_effects.items()
            except Exception:
                continue
            for inbc, ef in re_items:
                n_j = int(n_por_inbc.get(inbc, 0))
                if n_j == 0:
                    continue
                filas.append({
                    'inbc_id': inbc, 'competencia': comp, 'nbc_norm': nbc,
                    'VAE_propuesta': float(ef.iloc[0]),
                    'se_VAE_propuesta': float(np.sqrt(_var_posterior_blup(tau, sigma2, n_j))),
                })
        if i % 10 == 0:
            print(f"  propuesta: {i}/{len(nbcs)} NBC")
    return pd.DataFrame(filas), inse_col

# %% Batería comparativa
def _rango_intra_nbc(df, col):
    return df.groupby(['nbc_norm', 'competencia'])[col].rank(pct=True)


def bateria_comparativa(comp_df, medidas):
    out = {}

    filas = []
    for a, b in combinations(medidas, 2):
        for comp in sorted(comp_df['competencia'].unique()):
            g = comp_df[comp_df['competencia'] == comp].dropna(
                subset=[medidas[a], medidas[b]])
            if len(g) < 10:
                continue
            rho, p_rho = stats.spearmanr(g[medidas[a]], g[medidas[b]])
            tau_k, p_tau = stats.kendalltau(g[medidas[a]], g[medidas[b]])
            filas.append({'medida_A': a, 'medida_B': b, 'competencia': comp,
                          'n_INBC': len(g), 'spearman': rho, 'p_spearman': p_rho,
                          'kendall': tau_k, 'p_kendall': p_tau})
    out['concordancia'] = pd.DataFrame(filas)

    filas, matrices = [], {}
    for a, b in combinations(medidas, 2):
        for comp in sorted(comp_df['competencia'].unique()):
            g = comp_df[comp_df['competencia'] == comp].dropna(
                subset=[medidas[a], medidas[b]]).copy()
            if len(g) < 30:
                continue
            for etq, col in [('A', medidas[a]), ('B', medidas[b])]:
                g[f'tercil_{etq}'] = pd.qcut(g[col].rank(method='first'), 3,
                                             labels=[1, 2, 3]).astype(int)
            k = cohen_kappa_score(g['tercil_A'], g['tercil_B'], weights='quadratic')
            cambio = (g['tercil_A'] != g['tercil_B']).mean()
            extremo = ((g['tercil_A'] - g['tercil_B']).abs() == 2).mean()
            filas.append({'medida_A': a, 'medida_B': b, 'competencia': comp,
                          'n_INBC': len(g), 'kappa_ponderado': k,
                          'pct_cambia_tercil': cambio,
                          'pct_salto_extremo': extremo})
            matrices[f'{a}_vs_{b}_{comp}'] = pd.crosstab(
                g['tercil_A'], g['tercil_B'],
                rownames=[a], colnames=[b])
    out['terciles'] = pd.DataFrame(filas)
    out['matrices_transicion'] = matrices

    filas = []
    for etq, col in medidas.items():
        for comp in sorted(comp_df['competencia'].unique()):
            g = comp_df[comp_df['competencia'] == comp].dropna(
                subset=[col, 's11_global_medio'])
            if len(g) < 30:
                continue
            for pred_label, preds in [
                ('S11 medio', ['s11_global_medio']),
                ('S11 + INSE medios', ['s11_global_medio', 'inse_s11_medio']),
            ]:
                gg = g.dropna(subset=preds)
                if len(gg) < 30:
                    continue
                try:
                    mod = smf.ols(f"{col} ~ " + " + ".join(preds), data=gg).fit()
                    filas.append({'medida': etq, 'competencia': comp,
                                  'predictores': pred_label, 'n_INBC': len(gg),
                                  'R2': mod.rsquared, 'R2_ajustado': mod.rsquared_adj})
                except Exception as e:
                    print(f"    [sesgo] fallo {etq}/{comp}: {e}")
    out['sesgo_composicion'] = pd.DataFrame(filas)

    filas = []
    for etq, se_col, val_col in [
        ('espejo', 'se_VA', 'VA_espejo'),
        ('propuesta', 'se_VAE_propuesta', 'VAE_propuesta'),
    ]:
        if se_col not in comp_df.columns or val_col not in comp_df.columns:
            continue
        for comp in sorted(comp_df['competencia'].unique()):
            g = comp_df[comp_df['competencia'] == comp].dropna(subset=[se_col, val_col])
            if len(g) < 10:
                continue
            distinguible = (g[val_col].abs() > 1.96 * g[se_col]).mean()
            filas.append({'medida': etq, 'competencia': comp, 'n_INBC': len(g),
                          'pct_distinguible_del_promedio': distinguible,
                          'se_mediano': g[se_col].median()})
    out['incertidumbre'] = pd.DataFrame(filas)

    disc = []
    for comp in sorted(comp_df['competencia'].unique()):
        g = comp_df[comp_df['competencia'] == comp].dropna(
            subset=['VA_espejo', 'VAE_propuesta', 'spro_bruto_medio']).copy()
        if len(g) < 30:
            continue
        g['rango_espejo'] = _rango_intra_nbc(g, 'VA_espejo')
        g['rango_propuesta'] = _rango_intra_nbc(g, 'VAE_propuesta')
        g['rango_bruto'] = _rango_intra_nbc(g, 'spro_bruto_medio')
        g['desplazamiento'] = g['rango_propuesta'] - g['rango_bruto']
        disc.append(g)
    if disc:
        d = pd.concat(disc, ignore_index=True)
        d['inbc_anonimo'] = ('INBC-' +
                             d.groupby('inbc_id').ngroup().add(1).astype(str).str.zfill(4))
        out['discordantes'] = d.reindex(
            d['desplazamiento'].abs().sort_values(ascending=False).index)
    else:
        out['discordantes'] = pd.DataFrame()

    return out

# %% Análisis de sensibilidad
ESCENARIOS_SENSIBILIDAD = [
    ('base', {}),
    ('min_estudiantes=15', {'min_estudiantes_inbc': 15}),
    ('min_estudiantes=40', {'min_estudiantes_inbc': 40}),
    ('min_ies_por_nbc=3', {'min_ies_por_nbc': 3}),
    ('vecindad=8', {'max_vecinos': 8}),
    ('vecindad=25', {'max_vecinos': 25}),
    ('un_solo_anio', {'exigir_ambos_anios': False}),
]


def analisis_sensibilidad(df, periodo, base_va, base_ar, denominadores=None,
                          escenarios=ESCENARIOS_SENSIBILIDAD):
    ref_va = base_va[['inbc_id', 'competencia', 'VA_espejo']].rename(
        columns={'VA_espejo': 'VA_base'})
    ref_ar = base_ar[['inbc_id', 'competencia', 'pct_vecindad']].rename(
        columns={'pct_vecindad': 'pct_base'})

    filas = []
    for nombre, cambios in escenarios:
        if nombre == 'base':
            continue
        p = dict(PARAMS); p.update(cambios)
        d, _ = aplicar_filtros_icfes(df, periodo, p, denominadores, verbose=False)
        if d.empty:
            continue
        _, va = estimar_valor_agregado(d)
        if va.empty:
            continue
        vec = construir_vecindades(va, max_vecinos=p['max_vecinos'])
        _, ar = calcular_aporte_relativo(vec, va)

        m = (va.merge(ref_va, on=['inbc_id', 'competencia'], how='inner')
               .merge(ar[['inbc_id', 'competencia', 'pct_vecindad']],
                      on=['inbc_id', 'competencia'], how='left')
               .merge(ref_ar, on=['inbc_id', 'competencia'], how='left'))

        for comp in sorted(m['competencia'].unique()):
            g = m[m['competencia'] == comp]
            if len(g) < 10:
                continue
            rho_va, _ = stats.spearmanr(g['VA_espejo'], g['VA_base'])
            gg = g.dropna(subset=['pct_vecindad', 'pct_base'])
            rho_ar = (stats.spearmanr(gg['pct_vecindad'], gg['pct_base'])[0]
                      if len(gg) >= 10 else np.nan)
            filas.append({'escenario': nombre, 'competencia': comp,
                          'n_INBC_retenidos': len(g),
                          'n_INBC_escenario': va[va['competencia'] == comp].shape[0],
                          'spearman_VA_vs_base': rho_va,
                          'spearman_AR_vs_base': rho_ar})
        print(f"  sensibilidad '{nombre}': {va['inbc_id'].nunique()} INBC")
    return pd.DataFrame(filas)

# %% Ejecución
def ejecutar(periodo_objetivo=None, correr_sensibilidad=True):
    df = cargar_y_preparar()
    periodos = construir_periodos(df)
    if not periodos:
        raise SystemExit("No hay períodos bianuales construibles.")

    periodo = periodo_objetivo or periodos[-1]
    print(f"\nPeríodo analizado: {periodo[0]}-{periodo[1]}")

    denominadores = cargar_denominadores_spro(RUTA_SABERPRO_CRUDO, list(periodo))
    if denominadores is None:
        print("  Sin denominadores: se omite el filtro del 40%.")

    d, atricion = aplicar_filtros_icfes(df, periodo, PARAMS, denominadores)
    atricion.to_csv(f"{RUTA_SALIDA}/tabla_atricion_{periodo[0]}_{periodo[1]}.csv",
                    index=False)
    if d.empty:
        raise SystemExit("La muestra quedó vacía tras los filtros.")

    print("\n[4] Modelo nulo por NBC...")
    icc_df = estimar_modelo_nulo(d)
    icc_df['nbc_norm'] = icc_df['nbc_norm'].map(normalizar_nbc)

    ref_rows = []
    for nbc, (rc, lc, ing, n) in ICC_REFERENCIA_ICFES_2017_2018.items():
        for comp, val in [('razona_cuantitativo', rc), ('lectura_critica', lc),
                          ('ingles', ing)]:
            ref_rows.append({'nbc_norm': nbc, 'competencia': comp,
                             'ICC_icfes_2017_2018': val, 'n_ies_icfes': n})
    icc_cal = icc_df.merge(pd.DataFrame(ref_rows),
                           on=['nbc_norm', 'competencia'], how='left')
    icc_cal['dif_ICC'] = icc_cal['ICC'] - icc_cal['ICC_icfes_2017_2018']
    emparejados = icc_cal.dropna(subset=['ICC_icfes_2017_2018'])
    if len(emparejados) >= 5:
        rho, _ = stats.spearmanr(emparejados['ICC'],
                                 emparejados['ICC_icfes_2017_2018'])
        print(f"  Calibración ICC vs Icfes 2017-2018: rho={rho:.3f}, "
              f"EAM={emparejados['dif_ICC'].abs().mean():.4f}, "
              f"n={len(emparejados)} pares NBC-competencia")

    print("\n[5] Modelo-espejo de valor agregado...")
    coefs, va = estimar_valor_agregado(d)

    print("\n[6-7] Vecindades y aporte relativo...")
    vecindades = construir_vecindades(va)
    ar_matriz, ar_resumen = calcular_aporte_relativo(vecindades, va)

    print("\n[8] Re-estimación de la propuesta...")
    propuesta, inse_usado = estimar_propuesta(d)

    print("\n[9] Batería comparativa...")
    comp_df = ar_resumen.merge(propuesta[['inbc_id', 'competencia',
                                          'VAE_propuesta', 'se_VAE_propuesta']],
                               on=['inbc_id', 'competencia'], how='left')
    medidas = {
        'VA espejo (Icfes)': 'VA_espejo',
        'Aporte relativo (pct vecindad)': 'pct_vecindad',
        'VAE propuesta': 'VAE_propuesta',
        'Puntaje bruto Saber Pro': 'spro_bruto_medio',
    }
    resultados = bateria_comparativa(comp_df, medidas)

    sens = pd.DataFrame()
    if correr_sensibilidad:
        print("\n[10] Análisis de sensibilidad...")
        sens = analisis_sensibilidad(df, periodo, va, ar_resumen, denominadores)

    sufijo = f"{periodo[0]}_{periodo[1]}"
    salidas = {
        f'icc_por_nbc_{sufijo}': icc_cal,
        f'coeficientes_espejo_{sufijo}': coefs,
        f'va_espejo_por_inbc_{sufijo}': va,
        f'aporte_relativo_matriz_{sufijo}': ar_matriz,
        f'aporte_relativo_resumen_{sufijo}': ar_resumen,
        f'vae_propuesta_inbc_{sufijo}': propuesta,
        f'comparacion_completa_{sufijo}': comp_df,
        f'concordancia_{sufijo}': resultados['concordancia'],
        f'terciles_kappa_{sufijo}': resultados['terciles'],
        f'sesgo_composicion_{sufijo}': resultados['sesgo_composicion'],
        f'incertidumbre_{sufijo}': resultados['incertidumbre'],
        f'casos_discordantes_{sufijo}': resultados['discordantes'],
        f'sensibilidad_{sufijo}': sens,
    }
    for nombre, tabla in salidas.items():
        if isinstance(tabla, pd.DataFrame) and not tabla.empty:
            tabla.to_csv(f"{RUTA_SALIDA}/{nombre}.csv", index=False)
            print(f"  {nombre}.csv ({len(tabla)} filas)")

    for k, m in resultados['matrices_transicion'].items():
        seguro = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in k)
        m.to_csv(f"{RUTA_SALIDA}/matriz_{seguro}_{sufijo}.csv")

    with open(f"{RUTA_SALIDA}/parametros_{sufijo}.json", 'w', encoding='utf-8') as f:
        json.dump({'periodo': [int(p) for p in periodo], 'params': PARAMS,
                   'inse_utilizado': inse_usado,
                   'filtro_40pct_aplicado': denominadores is not None},
                  f, indent=2, ensure_ascii=False)

    graficar(comp_df, resultados, sufijo)
    resumen_ejecutivo(atricion, icc_cal, comp_df, resultados, sens, periodo, inse_usado)

    return {'datos': d, 'icc': icc_cal, 'coefs': coefs, 'va': va,
            'ar_resumen': ar_resumen, 'propuesta': propuesta,
            'comparacion': comp_df, 'resultados': resultados, 'sensibilidad': sens}

# %%
def graficar(comp_df, resultados, sufijo):
    comps = sorted(comp_df['competencia'].unique())
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4.5), squeeze=False)
    for ax, comp in zip(axes[0], comps):
        g = comp_df[comp_df['competencia'] == comp].dropna(
            subset=['VA_espejo', 'VAE_propuesta'])
        if g.empty:
            continue
        ax.scatter(g['VA_espejo'], g['VAE_propuesta'], s=14, alpha=0.5, c='#378ADD')
        lims = [min(g['VA_espejo'].min(), g['VAE_propuesta'].min()),
                max(g['VA_espejo'].max(), g['VAE_propuesta'].max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.axvline(0, color='black', linewidth=0.4)
        rho, _ = stats.spearmanr(g['VA_espejo'], g['VAE_propuesta'])
        ax.set_title(f"{comp}\nSpearman = {rho:.3f}")
        ax.set_xlabel('VA modelo-espejo')
        ax.set_ylabel('VAE propuesta')
    plt.tight_layout()
    plt.savefig(f"{RUTA_SALIDA}/fig_5_1_espejo_vs_propuesta_{sufijo}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    sesgo = resultados['sesgo_composicion']
    if not sesgo.empty:
        piv = sesgo[sesgo['predictores'] == 'S11 + INSE medios'].pivot_table(
            index='medida', columns='competencia', values='R2')
        if not piv.empty:
            ax = piv.plot(kind='barh', figsize=(9, 5), width=0.75)
            ax.set_xlabel('R² al regresar la medida sobre S11 e INSE medios de entrada')
            ax.set_ylabel('')
            ax.set_title('Diagnóstico de sesgo de composición')
            ax.legend(title='Competencia', fontsize=9)
            plt.tight_layout()
            plt.savefig(f"{RUTA_SALIDA}/fig_5_2_sesgo_composicion_{sufijo}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

    g = comp_df[comp_df['competencia'] == comps[0]].dropna(
        subset=['VA_espejo', 'se_VA']).sort_values('VA_espejo').reset_index(drop=True)
    if not g.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        signif = g['VA_espejo'].abs() > 1.96 * g['se_VA']
        ax.errorbar(range(len(g)), g['VA_espejo'],
                    yerr=1.96 * g['se_VA'], fmt='none',
                    ecolor='#CCCCCC', elinewidth=0.7)
        ax.scatter(range(len(g)), g['VA_espejo'], s=9,
                   c=np.where(signif, '#E24B4A', '#999999'))
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xlabel('INBC ordenadas por VA')
        ax.set_ylabel('VA modelo-espejo (escala del módulo)')
        ax.set_title(f'Incertidumbre del ordenamiento: {comps[0]}\n'
                     f'{signif.mean():.0%} de las INBC son distinguibles del promedio')
        plt.tight_layout()
        plt.savefig(f"{RUTA_SALIDA}/fig_5_3_incertidumbre_{sufijo}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  Figuras guardadas en {RUTA_SALIDA}/")

# %%
def resumen_ejecutivo(atricion, icc_cal, comp_df, resultados, sens, periodo, inse_usado):
    print()
    print(f"Resumen, período {periodo[0]}-{periodo[1]}")

    fin = atricion.iloc[-1]
    ini = atricion.iloc[0]
    print(f"\n1. MUESTRA: de {ini['estudiantes']:,} estudiantes y {ini['INBC']:,} INBC "
          f"se retienen {fin['estudiantes']:,} y {fin['INBC']:,} "
          f"({fin['IES']} IES en {fin['NBC']} NBC).")
    print(f"   Retención de estudiantes: {fin['estudiantes']/max(ini['estudiantes'],1):.1%}")

    conc = resultados['concordancia']
    if not conc.empty:
        print("\n2. CONCORDANCIA (Spearman) entre pares de medidas:")
        for _, r in conc.iterrows():
            print(f"   {r['medida_A']} vs {r['medida_B']} [{r['competencia']}]: "
                  f"rho={r['spearman']:.3f} (n={r['n_INBC']})")

    ter = resultados['terciles']
    if not ter.empty:
        print("\n3. CAMBIO DE DECISIÓN (terciles):")
        for _, r in ter.iterrows():
            print(f"   {r['medida_A']} vs {r['medida_B']} [{r['competencia']}]: "
                  f"{r['pct_cambia_tercil']:.1%} cambia de tercil, "
                  f"kappa={r['kappa_ponderado']:.3f}")

    ses = resultados['sesgo_composicion']
    if not ses.empty:
        print("\n4. SESGO DE COMPOSICIÓN (R² sobre S11 e INSE medios de entrada):")
        sub = ses[ses['predictores'] == 'S11 + INSE medios']
        for _, r in sub.iterrows():
            print(f"   {r['medida']} [{r['competencia']}]: R²={r['R2']:.3f}")

    inc = resultados['incertidumbre']
    if not inc.empty:
        print("\n5. INCERTIDUMBRE:")
        for _, r in inc.iterrows():
            print(f"   {r['medida']} [{r['competencia']}]: "
                  f"{r['pct_distinguible_del_promedio']:.1%} de INBC distinguibles "
                  f"del promedio de su NBC")

    if not sens.empty:
        print("\n6. SENSIBILIDAD A LOS UMBRALES (Spearman vs escenario base):")
        for _, r in sens.iterrows():
            print(f"   {r['escenario']} [{r['competencia']}]: "
                  f"rho(VA)={r['spearman_VA_vs_base']:.3f} | "
                  f"rho(AR)={r['spearman_AR_vs_base']:.3f} "
                  f"({r['n_INBC_escenario']} INBC)")

# %%
if __name__ == '__main__':
    salida = ejecutar()

# %%
