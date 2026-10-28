#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actualizador de Datos Climatológicos - Estación Cerro Negro (Corral Negro)
IANIGLA - CONICET / Observatorio Andino
"""

import os
import sys
import json
import math
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Configuración de URLs y directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_ESTACION = "https://observatorioandino.com/A003/"
CSV_FILE = os.path.join(BASE_DIR, "cerro_negro_datos.csv")
JSON_ACTUAL = os.path.join(BASE_DIR, "clima_actual.json")
JSON_RESUMEN = os.path.join(BASE_DIR, "resumen_datos.json")

def deg_to_cardinal(deg):
    if deg is None or math.isnan(deg):
        return "--"
    val = int((deg / 22.5) + 0.5)
    cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return cardinals[(val % 16)]

def calc_dew_point(temp, hum):
    """Calcula el punto de rocío usando la aproximación de Magnus-Tetens"""
    if temp is None or hum is None or math.isnan(temp) or math.isnan(hum) or hum <= 0:
        return None
    a = 17.27
    b = 237.7
    alpha = ((a * temp) / (b + temp)) + math.log(hum / 100.0)
    return round((b * alpha) / (a - alpha), 1)

def calc_wind_chill(temp, v_kmh):
    """Calcula sensación térmica / wind chill para temperaturas bajas y viento"""
    if temp is None or v_kmh is None or math.isnan(temp) or math.isnan(v_kmh):
        return None
    if temp <= 10.0 and v_kmh >= 4.8:
        # Fórmula internacional de Wind Chill
        wc = 13.12 + 0.6215 * temp - 11.37 * (v_kmh ** 0.16) + 0.3965 * temp * (v_kmh ** 0.16)
        return round(wc, 1)
    return round(temp, 1)

def descargar_con_playwright():
    """Descarga el CSV oficial desde Observatorio Andino interactuando con la aplicación Shiny"""
    print(f"1. Conectando con Observatorio Andino: {URL_ESTACION}...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page()
        page.set_default_timeout(60000)
        page.goto(URL_ESTACION, wait_until="domcontentloaded")

        print("2. Esperando botón de descarga en flexdashboard (#descarga_aws)...")
        page.wait_for_selector("#descarga_aws", timeout=45000)
        page.locator("#descarga_aws").click()

        print("3. Esperando enlace en ventana emergente (#DownloadData)...")
        page.wait_for_selector("#DownloadData", timeout=20000)
        dl_button = page.locator("#DownloadData")

        print("4. Iniciando descarga del archivo CSV...")
        temp_csv = os.path.join(BASE_DIR, "descarga_temp.csv")
        with page.expect_download(timeout=60000) as download_info:
            dl_button.click()
        
        download = download_info.value
        download.save_as(temp_csv)
        browser.close()

        if os.path.exists(temp_csv) and os.path.getsize(temp_csv) > 10000:
            if os.path.exists(CSV_FILE):
                try:
                    os.remove(CSV_FILE)
                except:
                    pass
            os.replace(temp_csv, CSV_FILE)
            print(f"-> Descarga completada: {os.path.getsize(CSV_FILE):,} bytes guardados en {CSV_FILE}")
            return True
        else:
            raise Exception("El archivo descargado está vacío o es inválido.")

def procesar_datos():
    print(f"5. Procesando dataset meteorológico: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"No se encontró el archivo {CSV_FILE}")

    col_names = [
        'fecha', 'dia_juliano', 'hora', 'temp_aire', 'hum_rel',
        'pres_atm', 'precipitacion', 'vel_viento', 'dir_viento',
        'irradiancia', 'temp_suelo', 'alt_nieve'
    ]

    df = pd.read_csv(
        CSV_FILE,
        skiprows=12,
        header=None,
        names=col_names,
        na_values=['NA', 'null', 'None', '-8190', '-9999', 'nan']
    )

    # Limpieza de fechas
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha']).sort_values('fecha').reset_index(drop=True)

    # Limpieza física de variables (control de calidad / rangos realistas)
    df.loc[(df['temp_aire'] < -40) | (df['temp_aire'] > 45), 'temp_aire'] = np.nan
    df.loc[(df['temp_suelo'] < -40) | (df['temp_suelo'] > 50), 'temp_suelo'] = np.nan
    df.loc[(df['hum_rel'] < 0) | (df['hum_rel'] > 100), 'hum_rel'] = np.nan
    df.loc[(df['pres_atm'] < 450) | (df['pres_atm'] > 750), 'pres_atm'] = np.nan
    df.loc[df['precipitacion'] < 0, 'precipitacion'] = 0.0
    df.loc[(df['vel_viento'] < 0) | (df['vel_viento'] > 70), 'vel_viento'] = np.nan
    df.loc[(df['dir_viento'] < 0) | (df['dir_viento'] > 360), 'dir_viento'] = np.nan
    df.loc[df['irradiancia'] < 0, 'irradiancia'] = 0.0

    # Nieve: pequeñas oscilaciones negativas por calibración ultrasónica en suelo pelado se truncan a 0
    df.loc[df['alt_nieve'] < 0, 'alt_nieve'] = 0.0
    df.loc[df['alt_nieve'] > 200, 'alt_nieve'] = np.nan

    # Viento en km/h
    df['vel_viento_kmh'] = (df['vel_viento'] * 3.6).round(1)

    print(f"   Total de registros válidos: {len(df):,}")

    # Última observación válida
    latest_row = df.iloc[-1]
    ts_latest = latest_row['fecha']
    print(f"   Última observación: {ts_latest.strftime('%Y-%m-%d %H:%M:%S')}")

    # Filtrar últimas 24h, 48h, 7d
    df_24h = df[df['fecha'] >= (ts_latest - timedelta(hours=24))]
    df_48h = df[df['fecha'] >= (ts_latest - timedelta(hours=48))]
    df_7d  = df[df['fecha'] >= (ts_latest - timedelta(days=7))]

    # Helpers para min/max con hora
    def get_min_max_info(sub_df, col):
        valid = sub_df.dropna(subset=[col])
        if len(valid) == 0:
            return {"min": None, "min_time": "--", "max": None, "max_time": "--", "avg": None}
        idx_min = valid[col].idxmin()
        idx_max = valid[col].idxmax()
        row_min = valid.loc[idx_min]
        row_max = valid.loc[idx_max]
        return {
            "min": round(float(row_min[col]), 1),
            "min_time": row_min['fecha'].strftime("%H:%M"),
            "max": round(float(row_max[col]), 1),
            "max_time": row_max['fecha'].strftime("%H:%M"),
            "avg": round(float(valid[col].mean()), 1)
        }

    stats_temp_24h = get_min_max_info(df_24h, 'temp_aire')
    stats_suelo_24h = get_min_max_info(df_24h, 'temp_suelo')
    stats_hum_24h = get_min_max_info(df_24h, 'hum_rel')
    stats_pres_24h = get_min_max_info(df_24h, 'pres_atm')

    # Viento 24h
    v_max_kmh = float(df_24h['vel_viento_kmh'].max()) if len(df_24h['vel_viento_kmh'].dropna()) > 0 else 0.0
    v_avg_kmh = float(df_24h['vel_viento_kmh'].mean()) if len(df_24h['vel_viento_kmh'].dropna()) > 0 else 0.0

    # Lluvia 24h
    lluvia_24h = round(float(df_24h['precipitacion'].sum()), 1) if len(df_24h['precipitacion'].dropna()) > 0 else 0.0
    lluvia_48h = round(float(df_48h['precipitacion'].sum()), 1) if len(df_48h['precipitacion'].dropna()) > 0 else 0.0

    # Radiación 24h
    sol_max = round(float(df_24h['irradiancia'].max()), 0) if len(df_24h['irradiancia'].dropna()) > 0 else 0.0

    # Altura de nieve actual y variación 24h
    nieve_actual = round(float(latest_row['alt_nieve']), 1) if not pd.isna(latest_row['alt_nieve']) else 0.0
    nieve_24h_max = round(float(df_24h['alt_nieve'].max()), 1) if len(df_24h['alt_nieve'].dropna()) > 0 else 0.0

    # Derivadas actuales
    cur_temp = float(latest_row['temp_aire']) if not pd.isna(latest_row['temp_aire']) else None
    cur_hum = float(latest_row['hum_rel']) if not pd.isna(latest_row['hum_rel']) else None
    cur_v_kmh = float(latest_row['vel_viento_kmh']) if not pd.isna(latest_row['vel_viento_kmh']) else None
    cur_dir = float(latest_row['dir_viento']) if not pd.isna(latest_row['dir_viento']) else None

    cur_dew = calc_dew_point(cur_temp, cur_hum)
    cur_sens = calc_wind_chill(cur_temp, cur_v_kmh)

    # Tendencia barométrica (últimas 3 horas)
    tendencia_presion = "Estable"
    if len(df) >= 4:
        pres_ahora = df.iloc[-1]['pres_atm']
        pres_3h_atras = df.iloc[-4]['pres_atm']
        if not pd.isna(pres_ahora) and not pd.isna(pres_3h_atras):
            diff = pres_ahora - pres_3h_atras
            if diff >= 1.0:
                tendencia_presion = "Subiendo (+{:.1f} mbar)".format(diff)
            elif diff <= -1.0:
                tendencia_presion = "Bajando ({:.1f} mbar)".format(diff)
            else:
                tendencia_presion = "Estable ({:+.1f} mbar)".format(diff)

    # Serie para index (últimas 72 horas para selector 24h / 48h / 72h)
    df_reciente = df_7d.copy()
    serie_horas = []
    for _, r in df_reciente.iterrows():
        serie_horas.append({
            "fecha": r['fecha'].strftime("%Y-%m-%d %H:%M"),
            "hora": r['fecha'].strftime("%H:%M"),
            "dia": r['fecha'].strftime("%d/%m"),
            "temp": round(float(r['temp_aire']), 1) if not pd.isna(r['temp_aire']) else None,
            "temp_suelo": round(float(r['temp_suelo']), 1) if not pd.isna(r['temp_suelo']) else None,
            "hum": round(float(r['hum_rel']), 0) if not pd.isna(r['hum_rel']) else None,
            "pres": round(float(r['pres_atm']), 1) if not pd.isna(r['pres_atm']) else None,
            "viento_kmh": round(float(r['vel_viento_kmh']), 1) if not pd.isna(r['vel_viento_kmh']) else None,
            "dir_deg": round(float(r['dir_viento']), 0) if not pd.isna(r['dir_viento']) else None,
            "irrad": round(float(r['irradiancia']), 0) if not pd.isna(r['irradiancia']) else None,
            "precip": round(float(r['precipitacion']), 1) if not pd.isna(r['precipitacion']) else 0.0,
            "nieve": round(float(r['alt_nieve']), 1) if not pd.isna(r['alt_nieve']) else 0.0,
        })

    # JSON 1: clima_actual.json
    clima_actual = {
        "estacion": {
            "nombre": "Cerro Negro (Corral Negro)",
            "codigo": "A003",
            "latitud": -24.55,
            "longitud": -66.24,
            "altitud": "4.050 msnm",
            "ubicacion": "Puna Salteña, Salta, Argentina",
            "institucion": "IANIGLA - CONICET / Observatorio Andino",
            "cita": "Datos libres y gratuitos citando a: IANIGLA-CONICET, https://observatorioandino.com/estaciones",
            "url_oficial": URL_ESTACION
        },
        "ultima_actualizacion": ts_latest.strftime("%Y-%m-%d %H:%M:%S"),
        "actual": {
            "temp_aire": round(cur_temp, 1) if cur_temp is not None else "--",
            "temp_suelo": round(float(latest_row['temp_suelo']), 1) if not pd.isna(latest_row['temp_suelo']) else "--",
            "hum_rel": int(round(cur_hum, 0)) if cur_hum is not None else "--",
            "pres_atm": round(float(latest_row['pres_atm']), 1) if not pd.isna(latest_row['pres_atm']) else "--",
            "tendencia_presion": tendencia_presion,
            "vel_viento_kmh": round(cur_v_kmh, 1) if cur_v_kmh is not None else "--",
            "vel_viento_ms": round(float(latest_row['vel_viento']), 2) if not pd.isna(latest_row['vel_viento']) else "--",
            "dir_viento_deg": int(round(cur_dir, 0)) if cur_dir is not None else 0,
            "dir_viento_cardinal": deg_to_cardinal(cur_dir),
            "irradiancia": int(round(float(latest_row['irradiancia']), 0)) if not pd.isna(latest_row['irradiancia']) else 0,
            "alt_nieve": nieve_actual,
            "sensacion_termica": cur_sens,
            "punto_rocio": cur_dew,
            "precipitacion_horaria": round(float(latest_row['precipitacion']), 1) if not pd.isna(latest_row['precipitacion']) else 0.0
        },
        "estadisticas_24h": {
            "temp_aire": stats_temp_24h,
            "temp_suelo": stats_suelo_24h,
            "hum_rel": stats_hum_24h,
            "pres_atm": stats_pres_24h,
            "viento_max_kmh": round(v_max_kmh, 1),
            "viento_avg_kmh": round(v_avg_kmh, 1),
            "lluvia_acumulada_24h": lluvia_24h,
            "lluvia_acumulada_48h": lluvia_48h,
            "irradiancia_max": sol_max,
            "nieve_max_24h": nieve_24h_max
        },
        "serie_reciente": serie_horas
    }

    with open(JSON_ACTUAL, "w", encoding="utf-8") as f:
        json.dump(clima_actual, f, ensure_ascii=False, indent=2)
    print(f"-> Generado {JSON_ACTUAL} ({os.path.getsize(JSON_ACTUAL):,} bytes)")

    # -------------------------------------------------------------
    # JSON 2: resumen_datos.json (HISTÓRICO Y CLIMATOLOGÍA)
    # -------------------------------------------------------------
    print("6. Calculando agregados históricos diarios y mensuales...")

    # Récords absolutos históricos
    records = {
        "temp_min": {
            "valor": round(float(df['temp_aire'].min()), 1),
            "fecha": df.loc[df['temp_aire'].idxmin()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "temp_max": {
            "valor": round(float(df['temp_aire'].max()), 1),
            "fecha": df.loc[df['temp_aire'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "suelo_min": {
            "valor": round(float(df['temp_suelo'].min()), 1),
            "fecha": df.loc[df['temp_suelo'].idxmin()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "suelo_max": {
            "valor": round(float(df['temp_suelo'].max()), 1),
            "fecha": df.loc[df['temp_suelo'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "viento_max_kmh": {
            "valor": round(float(df['vel_viento_kmh'].max()), 1),
            "fecha": df.loc[df['vel_viento_kmh'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "nieve_max_cm": {
            "valor": round(float(df['alt_nieve'].max()), 1),
            "fecha": df.loc[df['alt_nieve'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "pres_min": {
            "valor": round(float(df['pres_atm'].min()), 1),
            "fecha": df.loc[df['pres_atm'].idxmin()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "pres_max": {
            "valor": round(float(df['pres_atm'].max()), 1),
            "fecha": df.loc[df['pres_atm'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        },
        "irrad_max": {
            "valor": round(float(df['irradiancia'].max()), 0),
            "fecha": df.loc[df['irradiancia'].idxmax()]['fecha'].strftime("%Y-%m-%d %H:%M")
        }
    }

    # Agregación Diaria
    df['date_only'] = df['fecha'].dt.date
    daily_groups = df.groupby('date_only')

    diarios = []
    for day, grp in daily_groups:
        diarios.append({
            "fecha": day.strftime("%Y-%m-%d"),
            "t_min": round(float(grp['temp_aire'].min()), 1) if not pd.isna(grp['temp_aire'].min()) else None,
            "t_max": round(float(grp['temp_aire'].max()), 1) if not pd.isna(grp['temp_aire'].max()) else None,
            "t_med": round(float(grp['temp_aire'].mean()), 1) if not pd.isna(grp['temp_aire'].mean()) else None,
            "s_min": round(float(grp['temp_suelo'].min()), 1) if not pd.isna(grp['temp_suelo'].min()) else None,
            "s_max": round(float(grp['temp_suelo'].max()), 1) if not pd.isna(grp['temp_suelo'].max()) else None,
            "s_med": round(float(grp['temp_suelo'].mean()), 1) if not pd.isna(grp['temp_suelo'].mean()) else None,
            "hum_med": round(float(grp['hum_rel'].mean()), 0) if not pd.isna(grp['hum_rel'].mean()) else None,
            "pres_med": round(float(grp['pres_atm'].mean()), 1) if not pd.isna(grp['pres_atm'].mean()) else None,
            "lluvia": round(float(grp['precipitacion'].sum()), 1) if not pd.isna(grp['precipitacion'].sum()) else 0.0,
            "v_max": round(float(grp['vel_viento_kmh'].max()), 1) if not pd.isna(grp['vel_viento_kmh'].max()) else None,
            "v_med": round(float(grp['vel_viento_kmh'].mean()), 1) if not pd.isna(grp['vel_viento_kmh'].mean()) else None,
            "nieve_max": round(float(grp['alt_nieve'].max()), 1) if not pd.isna(grp['alt_nieve'].max()) else 0.0,
            "sol_max": round(float(grp['irradiancia'].max()), 0) if not pd.isna(grp['irradiancia'].max()) else 0.0
        })

    # Agregación Mensual
    df['year_month'] = df['fecha'].dt.to_period('M')
    monthly_groups = df.groupby('year_month')
    meses_nombres = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    mensuales = []
    for ym, grp in monthly_groups:
        y = int(ym.year)
        m = int(ym.month)
        
        # Días con lluvia (> 0.2 mm diarios)
        daily_in_month = grp.groupby(grp['fecha'].dt.date)['precipitacion'].sum()
        dias_lluvia = int((daily_in_month >= 0.2).sum())
        
        # Días con nieve (> 1 cm)
        daily_snow_in_month = grp.groupby(grp['fecha'].dt.date)['alt_nieve'].max()
        dias_nieve = int((daily_snow_in_month >= 1.0).sum())

        mensuales.append({
            "periodo": str(ym),
            "anio": y,
            "mes": m,
            "nombre_mes": meses_nombres.get(m, ""),
            "t_med": round(float(grp['temp_aire'].mean()), 1) if not pd.isna(grp['temp_aire'].mean()) else None,
            "t_min_abs": round(float(grp['temp_aire'].min()), 1) if not pd.isna(grp['temp_aire'].min()) else None,
            "t_max_abs": round(float(grp['temp_aire'].max()), 1) if not pd.isna(grp['temp_aire'].max()) else None,
            "s_med": round(float(grp['temp_suelo'].mean()), 1) if not pd.isna(grp['temp_suelo'].mean()) else None,
            "hum_med": round(float(grp['hum_rel'].mean()), 0) if not pd.isna(grp['hum_rel'].mean()) else None,
            "pres_med": round(float(grp['pres_atm'].mean()), 1) if not pd.isna(grp['pres_atm'].mean()) else None,
            "lluvia_total": round(float(grp['precipitacion'].sum()), 1) if not pd.isna(grp['precipitacion'].sum()) else 0.0,
            "dias_lluvia": dias_lluvia,
            "dias_nieve": dias_nieve,
            "nieve_max": round(float(grp['alt_nieve'].max()), 1) if not pd.isna(grp['alt_nieve'].max()) else 0.0,
            "v_med": round(float(grp['vel_viento_kmh'].mean()), 1) if not pd.isna(grp['vel_viento_kmh'].mean()) else None,
            "v_max": round(float(grp['vel_viento_kmh'].max()), 1) if not pd.isna(grp['vel_viento_kmh'].max()) else None
        })

    resumen_data = {
        "estacion": clima_actual["estacion"],
        "periodo_historico": {
            "inicio": df['fecha'].min().strftime("%Y-%m-%d"),
            "fin": df['fecha'].max().strftime("%Y-%m-%d"),
            "total_dias": len(diarios),
            "total_registros_horarios": len(df)
        },
        "records": records,
        "mensuales": mensuales,
        "diarios": diarios
    }

    with open(JSON_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(resumen_data, f, ensure_ascii=False, indent=2)
    print(f"-> Generado {JSON_RESUMEN} ({os.path.getsize(JSON_RESUMEN):,} bytes)")
    print("\n¡Proceso de actualización completado exitosamente!")

def main():
    skip_download = "--no-download" in sys.argv
    if not skip_download:
        try:
            descargar_con_playwright()
        except Exception as e:
            print(f"AVISO: Falló la descarga automática ({e}).")
            if os.path.exists(CSV_FILE):
                print("Se procederá utilizando el archivo CSV existente...")
            else:
                # Si no existe en CERRO FINAL, intentar copiar desde el directorio padre
                parent_csv = os.path.join(os.path.dirname(BASE_DIR), "cerro_negro_latest.csv")
                if os.path.exists(parent_csv):
                    print(f"Copiando copia previa desde {parent_csv}...")
                    import shutil
                    shutil.copyfile(parent_csv, CSV_FILE)
                else:
                    raise e
    procesar_datos()

if __name__ == "__main__":
    main()
