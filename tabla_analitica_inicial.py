import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

RUTA_CRUCES = "C:/Users/usuario/Desktop/DataICFES/datos/cruces/"
RUTA_SABER11 = "C:/Users/usuario/Desktop/DataICFES/datos/saber11/"
RUTA_SABERPRO = "C:/Users/usuario/Desktop/DataICFES/datos/saberpro/"
RUTA_OUTPUT = "C:/Users/usuario/Desktop/DataICFES/datos/output/"

COHORTES_SPRO = [2019, 2022, 2023, 2024]

S11_MIN_PERIODO = 20142

def paso1_cargar_cruces():
    print("PASO 1: Cargando archivos de cruce S11-SPro")

    cruces = []
    for f in sorted(os.listdir(RUTA_CRUCES)):
        if f.endswith('.txt') or f.endswith('.csv'):
            sep = ';'
            df = pd.read_csv(os.path.join(RUTA_CRUCES, f), sep=sep, low_memory=False)
            df['anio_spro'] = df['periodo_sbpro'].astype(str).str[:4].astype(int)
            df = df[df['anio_spro'].isin(COHORTES_SPRO)]
            if len(df) > 0:
                print(f"  {f}: {len(df):,} registros (años SPro: {df['anio_spro'].unique()})")
                cruces.append(df)

    if not cruces:
        print("No se encontraron archivos de cruce.")
        return None

    cruce_total = pd.concat(cruces, ignore_index=True)

    cruce_total = cruce_total[cruce_total['periodo_sb11'] >= S11_MIN_PERIODO]

    print(f"\n  Total después de filtro S11 >= {S11_MIN_PERIODO}: {len(cruce_total):,}")
    print(f"  Periodos S11: {sorted(cruce_total['periodo_sb11'].unique())}")
    print(f"  Periodos SPro: {sorted(cruce_total['periodo_sbpro'].unique())}")

    dupes_spro = cruce_total.duplicated(subset='estu_consecutivo_sbpro').sum()
    if dupes_spro > 0:
        print(f"  {dupes_spro} duplicados en consecutivo SPro. Manteniendo primer registro.")
        cruce_total = cruce_total.drop_duplicates(subset='estu_consecutivo_sbpro', keep='first')

    return cruce_total


def paso2_cargar_saber11(periodos_necesarios):
    print()
    print("PASO 2: Cargando bases de Saber 11")

    S11_VARS = [
        'estu_consecutivo', 'periodo',
        'punt_global', 'punt_lectura_critica', 'punt_matematicas',
        'punt_c_naturales', 'punt_sociales_ciudadanas', 'punt_ingles',
        'estu_genero', 'estu_fechanacimiento', 'estu_estudiante',
        'estu_inse_individual', 'estu_nse_individual',
        'fami_estratovivienda', 'fami_educacionmadre', 'fami_educacionpadre',
        'fami_tieneinternet', 'fami_tienecomputador', 'fami_tieneautomovil',
        'fami_tienelavadora', 'fami_personashogar', 'fami_cuartoshogar',
        'fami_numlibros', 'fami_trabajolabormadre', 'fami_trabajolaborpadre',
        'cole_naturaleza', 'cole_area_ubicacion', 'cole_bilingue',
        'cole_jornada', 'cole_caracter',
        'cole_cod_depto_ubicacion', 'cole_cod_mcpio_ubicacion',
    ]

    dfs = []
    for f in sorted(os.listdir(RUTA_SABER11)):
        if f.endswith('.txt') or f.endswith('.csv'):
            df = pd.read_csv(os.path.join(RUTA_SABER11, f), sep=';', low_memory=False)
            cols_available = [c for c in S11_VARS if c in df.columns]
            df = df[cols_available]
            print(f"  {f}: {len(df):,} registros, {len(cols_available)} variables")
            dfs.append(df)

    if not dfs:
        print("No se encontraron archivos de Saber 11.")
        return None

    s11 = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total S11 concatenado: {len(s11):,}")
    return s11


def paso3_cargar_saberpro():
    print()
    print("PASO 3: Cargando bases de Saber Pro")

    SPRO_VARS = [
        'estu_consecutivo', 'periodo',
        'punt_global',
        'mod_lectura_critica_punt', 'mod_razona_cuantitat_punt',
        'mod_competen_ciudada_punt', 'mod_comuni_escrita_punt',
        'mod_ingles_punt',
        'percentil_global',
        'inst_cod_institucion', 'inst_nombre_institucion',
        'inst_caracter_academico', 'inst_origen',
        'estu_snies_prgmacademico', 'estu_prgm_academico',
        'estu_nucleo_pregrado', 'estu_nivel_prgm_academico',
        'estu_metodo_prgm',
        'estu_inst_codmunicipio', 'estu_inst_departamento',
        'estu_genero', 'estu_inse_individual', 'estu_nse_individual',
        'estu_nse_ies',
        'fami_estratovivienda', 'fami_educacionmadre', 'fami_educacionpadre',
        'fami_ocupacionmadre', 'fami_ocupacionpadre',
        'fami_tieneautomovil', 'fami_tienecomputador', 'fami_tieneinternet',
        'fami_tienelavadora', 'fami_tienemotocicleta', 'fami_tieneserviciotv',
        'fami_trabajolabormadre', 'fami_trabajolaborpadre',
        'estu_horassemanatrabaja', 'estu_discapacidad',
        'estu_agregado',
    ]

    dfs = []
    for f in sorted(os.listdir(RUTA_SABERPRO)):
        if f.endswith('.txt') or f.endswith('.csv'):
            df = pd.read_csv(os.path.join(RUTA_SABERPRO, f), sep=';', low_memory=False)
            anio = df['periodo'].astype(str).str[:4].astype(int).iloc[0]
            if anio in COHORTES_SPRO:
                cols_available = [c for c in SPRO_VARS if c in df.columns]
                df = df[cols_available]
                print(f"  {f}: {len(df):,} registros, {len(cols_available)} variables")
                dfs.append(df)
            else:
                print(f"  {f}: Año {anio} fuera de las cohortes seleccionadas, omitido")

    if not dfs:
        print("No se encontraron archivos de Saber Pro.")
        return None

    spro = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total SPro concatenado: {len(spro):,}")
    return spro


def paso4_construir_tabla(cruce, s11, spro):
    print()
    print("PASO 4: Construyendo tabla analítica (3 JOINs)")

    s11_renamed = s11.rename(columns={
        c: f"s11_{c}" if c != 'estu_consecutivo' else c
        for c in s11.columns
    })

    spro_renamed = spro.rename(columns={
        c: f"spro_{c}" if c != 'estu_consecutivo' else c
        for c in spro.columns
    })

    print(f"\n  JOIN 1: Cruce ({len(cruce):,}) × SPro ({len(spro):,})")
    tabla = cruce.merge(
        spro_renamed,
        left_on='estu_consecutivo_sbpro',
        right_on='estu_consecutivo',
        how='inner'
    )
    print(f"    {len(tabla):,} registros después del JOIN")

    print(f"\n  JOIN 2: Tabla ({len(tabla):,}) × S11 ({len(s11):,})")
    tabla = tabla.merge(
        s11_renamed,
        left_on='estu_consecutivo_sb11',
        right_on='estu_consecutivo',
        how='inner',
        suffixes=('', '_s11_dup')
    )
    print(f"    {len(tabla):,} registros después del JOIN")

    tabla['cohorte_spro'] = tabla['periodo_sbpro'].astype(str).str[:4].astype(int)

    return tabla


def paso5_diagnostico(tabla):
    print()
    print("PASO 5: Diagnóstico de la tabla analítica")

    print(f"\nDIMENSIONES: {len(tabla):,} estudiantes × {len(tabla.columns)} variables")

    print(f"\nCOHORTES:")
    print(tabla['cohorte_spro'].value_counts().sort_index().to_string())

    if 'spro_inst_cod_institucion' in tabla.columns:
        n_ies = tabla['spro_inst_cod_institucion'].nunique()
        print(f"\nESTRUCTURA JERÁRQUICA:")
        print(f"  IES únicas: {n_ies}")
        if 'spro_estu_snies_prgmacademico' in tabla.columns:
            n_prog = tabla['spro_estu_snies_prgmacademico'].nunique()
            print(f"  Programas SNIES únicos: {n_prog}")

        ies_sizes = tabla.groupby('spro_inst_cod_institucion').size()
        print(f"  Estudiantes por IES: min={ies_sizes.min()}, "
              f"median={ies_sizes.median():.0f}, max={ies_sizes.max()}")
        print(f"  IES con 30 o más estudiantes: {(ies_sizes >= 30).sum()}")

    print(f"\nPUNTAJES:")
    for var_name, label in [
        ('s11_punt_global', 'S11 global'),
        ('spro_punt_global', 'SPro global'),
        ('spro_mod_lectura_critica_punt', 'SPro lectura crítica'),
        ('spro_mod_razona_cuantitat_punt', 'SPro razonamiento cuant.'),
    ]:
        if var_name in tabla.columns:
            s = pd.to_numeric(tabla[var_name], errors='coerce')
            print(f"  {label}: mean={s.mean():.1f}, std={s.std():.1f}, "
                  f"miss={s.isna().mean()*100:.1f}%")

    if 's11_punt_global' in tabla.columns and 'spro_punt_global' in tabla.columns:
        s11_g = pd.to_numeric(tabla['s11_punt_global'], errors='coerce')
        spro_g = pd.to_numeric(tabla['spro_punt_global'], errors='coerce')
        mask = s11_g.notna() & spro_g.notna()
        corr = s11_g[mask].corr(spro_g[mask])
        print(f"\nCORRELACIÓN S11-SPro (puntaje global): r = {corr:.4f}")

    print(f"\nVARIABLES CON >10% MISSING:")
    for c in tabla.columns:
        miss = tabla[c].isna().mean() * 100
        if miss > 10:
            print(f"  {c}: {miss:.1f}%")

    if 'spro_inst_cod_institucion' in tabla.columns and 'spro_punt_global' in tabla.columns:
        y = pd.to_numeric(tabla['spro_punt_global'], errors='coerce')
        g = tabla['spro_inst_cod_institucion']
        mask = y.notna()
        grand_mean = y[mask].mean()
        ss_total = ((y[mask] - grand_mean) ** 2).sum()
        group_means = y[mask].groupby(g[mask]).transform('mean')
        ss_between = ((group_means - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        print(f"\nICC PROXY (η² bruto, SPro global entre IES): {eta_sq:.4f} ({eta_sq*100:.1f}%)")

    return tabla


def paso6_guardar(tabla):
    print()
    print("PASO 6: Guardando tabla analítica")

    os.makedirs(RUTA_OUTPUT, exist_ok=True)
    output_path = os.path.join(RUTA_OUTPUT, "tabla_analitica_VAE.csv")
    tabla.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Guardado en: {output_path}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"  Filas: {len(tabla):,}")
    print(f"  Columnas: {len(tabla.columns)}")


if __name__ == "__main__":
    print("Construcción de la tabla analítica")
    print("   Cohortes SPro:", COHORTES_SPRO)
    print("   Filtro S11: periodo >=", S11_MIN_PERIODO)
    print()

    cruce = paso1_cargar_cruces()
    if cruce is None:
        sys.exit(1)

    s11 = paso2_cargar_saber11(cruce['periodo_sb11'].unique())
    if s11 is None:
        sys.exit(1)

    spro = paso3_cargar_saberpro()
    if spro is None:
        sys.exit(1)

    tabla = paso4_construir_tabla(cruce, s11, spro)
    tabla = paso5_diagnostico(tabla)
    paso6_guardar(tabla)
