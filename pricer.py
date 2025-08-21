import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.interpolate import interp1d
import plotly.express as px
import numpy as np
import io
import chardet
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction robuste pour lire un CSV avec détection d'encodage
def try_read_csv(file, skiprows=0, sep_override=None):
    raw = file.read()
    file.seek(0)
    result = chardet.detect(raw)
    enc = result['encoding'] or 'utf-8'
    confidence = result['confidence']
    logger.info(f"Detected encoding: {enc}, confidence: {confidence}")

    encodings_to_try = [enc, 'latin-1', 'cp1252', 'iso-8859-1']
    if sep_override:
        seps = [sep_override]
    else:
        seps = [',', ';', '\t']

    for sep in seps:
        for encoding in encodings_to_try:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, encoding=encoding, skiprows=skiprows)
                logger.info(f"Successfully read file with encoding: {encoding}, separator: '{sep}'")
                st.write(f"Successfully read file with encoding: {encoding}, separator: '{sep}'")
                return df
            except UnicodeDecodeError as e:
                logger.warning(f"UnicodeDecodeError with {encoding} and {sep}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error with encoding {encoding} and separator '{sep}': {str(e)}")
                continue
    st.error("Impossible de lire le CSV. Vérifiez l'encodage et le séparateur. Essayez de convertir le fichier en UTF-8.")
    logger.error("Failed to read CSV after trying multiple encodings and separators.")
    st.stop()


# Fonction utilitaire pour extraire un scalaire d'un DataFrame, Series ou ndarray
def get_scalar(val):
    if hasattr(val, 'values'):
        if hasattr(val.values, 'size') and val.values.size > 0:
            return val.values[0]
        else:
            return val
    return val

# Modified CourbeTaux to support ZC curves, extrapolation, and conversion
class CourbeTaux:
    def __init__(self, df, date_valo: datetime, type='monetaire'):
        self.df = df.copy()
        self.type = type  # 'monetaire', 'actuariel', ou 'zc'
        self.date_valo = date_valo
        self._preparer_courbe()

    def _preparer_courbe(self):
        self.df['MATURITE_JOURS'] = (pd.to_datetime(self.df['Echeance']) - self.date_valo).dt.days
        self.df = self.df[self.df['MATURITE_JOURS'] > 0]
        self.df = self.df.sort_values(by='MATURITE_JOURS')
        maturites = self.df['MATURITE_JOURS']
        taux = self.df['Taux']
        self.interpolateur = interp1d(maturites, taux, kind='linear', fill_value='extrapolate')

    def get(self, maturite, cible='monetaire'):
        try:
            taux = float(self.interpolateur(maturite))
            if self.type != cible:
                return self.convertir(taux, maturite, self.type, cible)
            return taux
        except ValueError as e:
            logger.error(f"Interpolation error for maturite {maturite}: {str(e)}")
            return 0.0  # Fallback value

    def convertir(self, taux, maturite, source, cible):
        base = 365
        if source == cible:
            return taux
        if source == 'actuariel' and cible == 'monetaire':
            return ((1 + taux) ** (maturite / base) - 1) * (360 / maturite)
        elif source == 'monetaire' and cible == 'actuariel':
            return (1 + (taux * maturite / 360)) ** (base / maturite) - 1
        return taux  # No conversion for 'zc' assumed

# ObligationUCM Class
class ObligationUCM:
    def __init__(self, nominal, taux_facial, date_emission, date_echeance, date_valo,
                 frequence, type_taux='fixe', amortissable=False, revisable=False,
                 revision_freq=None, spread=0.0):
        self.nominal = nominal
        self.taux_facial = taux_facial
        self.date_emission = date_emission
        self.date_echeance = date_echeance
        self.date_valo = date_valo
        self.frequence = frequence  # Annual=12, Semestrial=6, Trimestrial=4, Monthly=1
        self.type_taux = type_taux
        self.amortissable = amortissable
        self.revisable = revisable
        self.revision_freq = revision_freq
        self.spread = spread
        self.period_days = 360 / (12 / frequence)

    def interpolate_rate(self, time):
        times = sorted(self.rate_curve.keys())
        if time <= times[0]:
            return self.rate_curve[times[0]]
        if time >= times[-1]:
            return self.rate_curve[times[-1]]
        for i in range(len(times)-1):
            if times[i] <= time < times[i+1]:
                t1, t2 = times[i], times[i+1]
                r1, r2 = self.rate_curve[t1], self.rate_curve[t2]
                return round((r1 + (r2 - r1) * (time - t1) / (t2 - t1)), 5)

    def generate_flux(self):
        fluxes = []
        delta = (self.date_echeance - self.date_valo).days / 360.0
        periods = int(delta * (12 / self.frequence))
        residual = delta * (12 / self.frequence) - periods

        if self.revisable and self.revision_freq:
            revision_periods = int(delta * self.revision_freq)
            for p in range(revision_periods + 1):
                t = p / self.revision_freq
                rate = self.interpolate_rate(t) + self.spread
                if t >= 1:
                    flux = self.nominal * (rate / (12 / self.revision_freq))
                else:
                    flux = self.nominal * (rate * self.period_days / 360)
                fluxes.append((t, flux))
            return fluxes

        if self.amortissable:
            for p in range(periods + 1):
                t = p / (12 / self.frequence)
                if self.frequence == 12:  # Monthly
                    if t >= 1:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) / 12
                    else:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) * (delta / 30)
                elif self.frequence == 4:  # Trimestrial
                    if t >= 1:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) / 4
                    else:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) * (delta / 90)
                elif self.frequence == 6:  # Semestrial
                    if t >= 1:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) / 6
                    else:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) * (delta / 180)
                elif self.frequence == 1:  # Annual
                    if t >= 1:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) / 1
                    else:
                        flux = self.nominal / (periods + 1) + self.nominal * (self.interpolate_rate(t) + self.spread) * (delta / 360)
                fluxes.append((t, flux))
        else:
            for p in range(periods + 1):
                t = p / (12 / self.frequence)
                if self.frequence == 12:  # Monthly
                    coupon = self.nominal * self.taux_facial * (30 / 360)
                    if t == delta:
                        flux = self.nominal + coupon
                    else:
                        flux = coupon
                elif self.frequence == 4:  # Trimestrial
                    coupon = self.nominal * self.taux_facial * (90 / 360)
                    if t == delta:
                        flux = self.nominal + coupon
                    else:
                        flux = coupon
                elif self.frequence == 6:  # Semestrial
                    coupon = self.nominal * self.taux_facial * (180 / 360)
                    if t == delta:
                        flux = self.nominal + coupon
                    else:
                        flux = coupon
                elif self.frequence == 1:  # Annual
                    coupon = self.nominal * self.taux_facial * (360 / 360)
                    if t == delta:
                        flux = self.nominal + coupon
                    else:
                        flux = coupon
                fluxes.append((t, flux))
        return fluxes

# Helper: Calculate precise f (fraction of year to next coupon)
def compute_f(date_valo, next_coupon):
    prior_coupon = next_coupon.replace(year=next_coupon.year - 1)
    return (next_coupon - date_valo).days / (next_coupon - prior_coupon).days

# Updated ObligationUCM.actualiser to fully support UCM Guide logic
def actualiser(self, courbe: CourbeTaux):
    self.rate_curve = dict(zip((pd.to_datetime(courbe.df['Echeance']) - self.date_valo).dt.days / 360.0, courbe.df['Taux']))
    total = 0.0
    fluxes = self.generate_flux()
    for t, flux in fluxes:
        time_days = t * self.period_days
        if time_days <= 365:
            tm = courbe.get(time_days, cible='monetaire')
            discount = 1 / (1 + (tm + self.spread) * time_days / 360)
        else:
            ta = courbe.get(time_days, cible='actuariel')
            if courbe.type == 'zc':
                discount = math.exp(-ta * (t + 1e-6))  # ZC exponential discount
            else:
                discount = 1 / ((1 + (ta + self.spread) / (12 / self.frequence)) ** (t * (12 / self.frequence)))
        total += flux * discount
    return round(total, 2)

# Patch the method into ObligationUCM class
from types import MethodType
ObligationUCM.actualiser = MethodType(actualiser, ObligationUCM)

# Streamlit Dashboard
st.title("Bond Valuation Dashboard")

# Sidebar for inputs
st.sidebar.header("Bond Parameters")

# Bond input fields
nominal = st.sidebar.number_input("Nominal Amount", min_value=0.0, value=100000.0, step=1000.0)
taux_facial = st.sidebar.number_input("Coupon Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
def get_single_date(val):
    if isinstance(val, (tuple, list)) and val:
        return val[0]
    return val

def safe_timestamp(val):
    if val is None:
        return pd.Timestamp('today')
    return pd.Timestamp(val)

date_emission = get_single_date(st.sidebar.date_input("Emission Date", value=datetime(2023, 1, 1).date()))
date_echeance = get_single_date(st.sidebar.date_input("Maturity Date", value=datetime(2030, 1, 1).date()))
date_valo = get_single_date(st.sidebar.date_input("Valuation Date", value=datetime(2025, 7, 21, 12, 22).date()))
frequence = st.sidebar.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12], index=3)
type_taux = st.sidebar.selectbox("Rate Type", ["fixe", "variable"], index=0)
amortissable = st.sidebar.checkbox("Amortizing Bond", value=False)
revisable = st.sidebar.checkbox("Revisable Bond", value=False)
revision_freq = st.sidebar.number_input("Revision Frequency (Years)", min_value=0, value=1, step=1) if revisable else None
spread = st.sidebar.number_input("Spread (%)", min_value=0.0, value=0.0, step=0.01) / 100

# Yield curve input
st.sidebar.header("Yield Curve Input")
yield_curve_option = st.sidebar.radio("Yield Curve Source", ["Upload Excel/CSV", "Manual Input"])

if yield_curve_option == "Upload Excel/CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Yield Curve Excel or CSV", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        if uploaded_file.name.endswith(('.csv')):
            df_yield = try_read_csv(uploaded_file)
        else:
            df_yield = pd.read_excel(uploaded_file)
        st.write("Uploaded Yield Curve Data:", df_yield)
    else:
        st.warning("Please upload an Excel or CSV file with columns 'Echeance' (date) and 'Taux' (%).")
        df_yield = pd.DataFrame()
else:
    st.sidebar.subheader("Manual Yield Curve Input")
    n_points = st.sidebar.number_input("Number of Yield Curve Points", min_value=1, value=3, step=1)
    dates = []
    rates = []
    for i in range(n_points):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            date = get_single_date(st.date_input(f"Maturity Date {i+1}", value=(datetime(2025, 7, 21, 12, 22) + timedelta(days=365*(i+1))).date()))
        with col2:
            rate = st.number_input(f"Rate {i+1} (%)", min_value=0.0, value=3.0 + i*0.5, step=0.1)
        dates.append(date)
        rates.append(rate)
    df_yield = pd.DataFrame({"Echeance": dates, "Taux": rates})
    st.write("Manual Yield Curve Data:", df_yield)

# Bond/prices Excel upload (multiple bonds)
st.sidebar.header("Batch Bond Pricing")
batch_bond_file = st.sidebar.file_uploader("Upload Excel file with bonds/prices (one per row)", type=["xlsx", "xls"])
batch_bond_df = None
if batch_bond_file:
    batch_bond_df = pd.read_excel(batch_bond_file)
    st.write("Uploaded Bonds Data:", batch_bond_df)

# Yield curve type
curve_type = st.sidebar.selectbox("Yield Curve Type", ["monetaire", "actuariel", "zc"], index=0)

# Main content
st.header("Bond Valuation Results")

# If batch bond file is uploaded, process all bonds in the file
def process_batch_bonds(batch_bond_df, df_yield, date_valo, curve_type):
    results = []
    # Try to auto-detect columns
    for idx, row in batch_bond_df.iterrows():
        try:
            nominal = row.get('Nominal', row.get('nominal', 100000.0))
            taux_facial = row.get('Coupon', row.get('coupon', 0.05))
            if taux_facial > 1:  # If given as percent
                taux_facial = taux_facial / 100
            date_emission = pd.to_datetime(row.get('Emission', row.get('date_emission', None)))
            date_echeance = pd.to_datetime(row.get('Echeance', row.get('date_echeance', None)))
            frequence = int(row.get('Frequence', row.get('frequence', 12)))
            type_taux = row.get('TypeTaux', row.get('type_taux', 'fixe'))
            amortissable = bool(row.get('Amortissable', row.get('amortissable', False)))
            revisable = bool(row.get('Revisable', row.get('revisable', False)))
            revision_freq = row.get('RevisionFreq', row.get('revision_freq', None))
            spread = row.get('Spread', row.get('spread', 0.0))
            if spread > 1:  # If given as percent
                spread = spread / 100
            # Prepare yield curve
            courbe = CourbeTaux(df_yield, date_valo, type=curve_type)
            bond = ObligationUCM(
                nominal=nominal,
                taux_facial=taux_facial,
                date_emission=date_emission,
                date_echeance=date_echeance,
                date_valo=date_valo,
                frequence=frequence,
                type_taux=type_taux,
                amortissable=amortissable,
                revisable=revisable,
                revision_freq=revision_freq,
                spread=spread
            )
            bond_value = bond.actualiser(courbe)
            results.append({
                'Nominal': nominal,
                'Coupon': taux_facial,
                'Emission': date_emission,
                'Echeance': date_echeance,
                'Value': bond_value
            })
        except Exception as e:
            logger.error(f"Batch bond processing error at index {idx}: {str(e)}")
            results.append({'Error': str(e)})
    return pd.DataFrame(results)

# If batch bond file is uploaded, process and show results
df_batch_results = None
if batch_bond_df is not None and not df_yield.empty:
    try:
        date_valo_batch = get_scalar(safe_timestamp(date_valo))
        if pd.isna(date_valo_batch):
            st.error("Valuation date is invalid.")
            st.stop()
        df_batch_results = process_batch_bonds(batch_bond_df, df_yield, date_valo_batch, curve_type)
        st.subheader("Batch Bond Valuation Results")
        st.dataframe(df_batch_results)
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        st.error(f"Batch pricing error: {str(e)}")

# If not batch, do single bond pricing as before
if not df_yield.empty and (batch_bond_df is None):
    try:
        date_emission = safe_timestamp(date_emission)
        date_echeance = safe_timestamp(date_echeance)
        date_valo = safe_timestamp(date_valo)
        if pd.isna(date_valo):
            st.error("Valuation date is invalid.")
            st.stop()

        courbe = CourbeTaux(df_yield, date_valo, type=curve_type)

        bond = ObligationUCM(
            nominal=nominal,
            taux_facial=taux_facial,
            date_emission=date_emission,
            date_echeance=date_echeance,
            date_valo=date_valo,
            frequence=frequence,
            type_taux=type_taux,
            amortissable=amortissable,
            revisable=revisable,
            revision_freq=revision_freq,
            spread=spread
        )

        bond_value = bond.actualiser(courbe)
        st.subheader(f"Bond Value: {bond_value:,.2f}")

        flux = pd.DataFrame(bond.generate_flux(), columns=['Time (Years)', 'Flux'])
        st.subheader("Cash Flows")
        st.dataframe(flux)

        st.subheader("Yield Curve")
        df_yield['MATURITE_JOURS'] = (pd.to_datetime(df_yield['Echeance']) - date_valo).dt.days
        df_yield = df_yield[df_yield['MATURITE_JOURS'] > 0].sort_values(by=['MATURITE_JOURS'])
        fig = px.line(df_yield, x='MATURITE_JOURS', y='Taux', title="Yield Curve", labels={"MATURITE_JOURS": "Days to Maturity", "Taux": "Rate (%)"})
        st.plotly_chart(fig)

    except Exception as e:
        logger.error(f"Single bond valuation error: {str(e)}")
        st.error(f"Error: {str(e)}")
else:
    if batch_bond_df is None:
        st.error("Please provide valid yield curve data to proceed.")

# === NOUVEAU WORKFLOW POUR TRAITEMENT DE FICHIER OBLIGATIONS ===
st.title("Valorisation Obligations par Courbe de Taux Actuarielle")

st.header("1. Charger la courbe des taux actuarielle (CSV)")
st.info("Le fichier CSV doit contenir une colonne de dates d'échéance (maturité) et une colonne de taux. Les noms de colonnes peuvent être libres, mais il est conseillé d'utiliser 'Echeance' et 'Taux'.")
curve_file = st.file_uploader("Fichier CSV de la courbe des taux (actuarielle)", type=["csv"])
df_curve = None
if curve_file:
    df_curve = try_read_csv(curve_file, skiprows=2, sep_override=';')
    df_curve = df_curve.loc[:, ~df_curve.columns.duplicated()]
    df_curve.columns = [col.replace('"', '').strip() for col in df_curve.columns]
    st.write(f"Colonnes détectées dans le CSV : {df_curve.columns.tolist()}")

    date_col = "Date d'échéance"
    taux_col = "Taux moyen pondéré"

    if date_col not in df_curve.columns or taux_col not in df_curve.columns:
        st.error(f"Colonnes attendues non trouvées. Colonnes détectées : {df_curve.columns.tolist()}")
        st.stop()

    df_curve = df_curve[[date_col, taux_col]].rename(columns={date_col: 'Echeance', taux_col: 'Taux'})
    df_curve = df_curve[pd.to_datetime(df_curve['Echeance'], format='%d/%m/%Y', errors='coerce').notna()]
    df_curve['Echeance'] = pd.to_datetime(df_curve['Echeance'], format='%d/%m/%Y', dayfirst=True)
    df_curve['Taux'] = (
        df_curve['Taux']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip()
        .astype(float)
    )
    st.write("Aperçu courbe des taux :", df_curve)

st.header("2. Charger le fichier des obligations à valoriser (Excel)")
bonds_file = st.file_uploader("Fichier Excel des obligations à valoriser", type=["xlsx", "xls"], key="bonds")
df_bonds = None
if bonds_file:
    df_bonds = pd.read_excel(bonds_file)
    st.write("Aperçu obligations :", df_bonds.head())

def valoriser_obligation(row, courbe, date_valo):
    try:
        nominal = row['NOMINAL']
        taux_facial = row['VALEUR_TAUX']
        if taux_facial > 1:
            taux_facial = taux_facial / 100
        date_emission = pd.to_datetime(row['DATE_EMISSION'], format='%Y-%m-%d')
        date_echeance = pd.to_datetime(row['DATE_ECHEANCE'], format='%Y-%m-%d')
        date_valo = pd.to_datetime(get_scalar(date_valo), format='%Y-%m-%d')
        freq_map = {'AN': 12, 'SEM': 6, 'TRI': 3, 'MENS': 1}
        raw_freq = str(row.get('PERIODICITE_COUPON', 12)).upper()
        frequence = freq_map.get(raw_freq, 12)
        type_taux = str(row.get('TYPE_TAUX', 'fixe')).lower()
        spread = row.get('SPREAD_EMISSION', 0.0)
        if spread > 1:
            spread = spread / 100

        time_to_maturity = (date_echeance - date_valo).days
        if time_to_maturity <= 0:
            return nominal

        courbe_obj = CourbeTaux(df=courbe, date_valo=date_valo, type='actuariel')
        bond = ObligationUCM(
            nominal=nominal,
            taux_facial=taux_facial,
            date_emission=date_emission,
            date_echeance=date_echeance,
            date_valo=date_valo,
            frequence=frequence,
            type_taux=type_taux,
            amortissable=False,
            revisable=False,
            revision_freq=None,
            spread=spread
        )
        return bond.actualiser(courbe_obj)
    except Exception as e:
        logger.error(f"Valuation error for row {row.name}: {str(e)}")
        return f"Erreur: {e}"

if df_curve is not None and df_bonds is not None:
    st.header("3. Calcul de la valorisation des obligations")
    date_valo = pd.to_datetime(df_curve['Echeance']).min()
    st.info("Calcul en cours...")
    df_bonds['VALORISATION'] = df_bonds.apply(lambda row: valoriser_obligation(row, df_curve, date_valo), axis=1)
    df_result = df_bonds[['CODE', 'CODE_ISIN', 'VALORISATION']]
    st.success("Valorisation terminée !")
    st.dataframe(df_result)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False)
    output.seek(0)
    st.download_button(
        label="Télécharger le fichier de valorisation (Excel)",
        data=output,
        file_name="valorisation_obligations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


