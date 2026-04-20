# Analisis Data Iklim: Korelasi Hujan-ENSO di Benua Maritim

Repositori ini berisi analisis mendalam tentang hubungan antara pola curah hujan dan fenomena El Niño Southern Oscillation (ENSO) di kawasan Benua Maritim menggunakan berbagai metode analisis statistik dan klimatologi.

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Struktur Direktori](#struktur-direktori)
- [Deskripsi Analisis](#deskripsi-analisis)
- [Data yang Digunakan](#data-yang-digunakan)
- [Persyaratan Lingkungan](#persyaratan-lingkungan)
- [Cara Menggunakan](#cara-menggunakan)
  - [🔑 Konfigurasi Path Data](#-konfigurasi-path-data-penting)
- [Keluaran dan Hasil](#keluaran-dan-hasil)
- [Referensi](#referensi)
- [ℹ️ Dokumentasi Tambahan](#ℹ️-dokumentasi-tambahan)
- [🔐 Notebook Output Management](#-notebook-output-management)

## 🎯 Tentang Proyek

Proyek ini merupakan bagian dari penelitian tugas akhir (skripsi) yang fokus pada:

1. **Analisis Korelasi Spasial-Temporal**: Mengidentifikasi bagaimana variasi ENSO (diwakili oleh Indeks Niño3.4) mempengaruhi pola curah hujan di berbagai domain geografis
2. **Analisis EOF (Empirical Orthogonal Function)**: Menemukan mode variabilitas utama dalam pola curah hujan musiman
3. **Studi Non-stasioneritas**: Menyelidiki perubahan dalam kekuatan hubungan hujan-ENSO sepanjang waktu
4. **Analisis Sirkulasi Atmosfer**: Mengkaji mekanisme dinamika udara 850 hPa dan pola sirkular yang terlibat
5. **Analisis ITCZ (Intertropical Convergence Zone)**: Memahami pergeseran dan intensitas zona konvergensi tropika

## 📁 Struktur Direktori

```
data_processing/notebooks/
├── comprehensive_analysis/      # Analisis komprehensif korelasi hujan-ENSO
│   ├── scripts/                 # Script Python untuk generate notebook
│   ├── correlation_mc.ipynb     # Notebook analisis korelasi Maritime Continent
│   ├── rainfall_analysis_v3.ipynb # Analisis curah hujan
│   ├── wind_analysis_v3.ipynb   # Analisis angin dan sirkulasi
│   ├── mfc_analysis_v2.ipynb    # Analisis moisture flux convergence
│   └── data_nc/                 # Output data NetCDF
│
├── divided_correlation/         # Analisis korelasi terbagi (periode/domain)
│   └── dcorr_v6_workflow.ipynb  # Workflow utama divided correlation
│
├── eof_analysis/               # Analisis Empirical Orthogonal Function
│   ├── build_mswep_eof_mc_notebook.py # Script generator notebook
│   └── README.md               # Dokumentasi khusus EOF
│
├── non_stationarity/           # Analisis non-stasioneritas hubungan hujan-ENSO
│   ├── PLAN.md                 # Rencana penelitian
│   └── PLAN-non-stationarity.md
│
├── lagged_correlation/         # Analisis korelasi tertinggal (lag analysis)
│   # Menyimpan hasil korelasi pada berbagai lag waktu
│
├── running_correlation/        # Analisis korelasi bergerak (moving correlation)
│   └── djf_runningcorr_domainjson_layoutAC.py
│
├── itcz_analysis/             # Analisis ITCZ (Intertropical Convergence Zone)
│   ├── itcz_analysis_v6_gpcp_nosmooth.ipynb
│   ├── itcz_wind.ipynb
│   ├── itcz_wind_clim.ipynb
│   └── *.png                  # Visualisasi hasil analisis
│
├── circulation850hpa/         # Analisis sirkulasi di level 850 hPa
│   # Analisis medan tekanan dan angin
│
├── cluster_enso/              # Analisis pengelompokan fase ENSO
│   └── v2_generate_combined_notebook.py
│
└── README.md                  # File ini
```

## 📊 Deskripsi Analisis

### 1. **Comprehensive Analysis** - Analisis Komprehensif
Analisis lengkap yang mengintegrasikan curah hujan, angin, dan indeks ENSO. Meliputi:
- Komputasi rata-rata seasonal DJF (Desember-Januari-Februari)
- Perhitungan korelasi spasial terhadap Niño3.4
- Analisis signifikansi statistik
- Visualisasi korelasi global dan regional
- **Data Output**: NetCDF dengan medan korelasi dan p-value

**File Utama:**
- `build_correlation_global.py`: Komputasi korelasi di seluruh domain
- `build_correlation_mc_v3.py`: Fokus pada Maritime Continent
- `rainfall_analysis_v3.ipynb`: Eksplorasi pola curah hujan
- `wind_analysis_v3.ipynb`: Analisis dinamika angin
- `mfc_analysis_v2.ipynb`: Analisis konvergensi moisture flux

### 2. **EOF Analysis** - Analisis Mode Utama Variabilitas
Mengidentifikasi mode variabilitas utama dalam curah hujan seasonal menggunakan PCA:
- Fokus pada Benua Maritim yang dibagi menjadi WMC dan EMC
- Analisis komponen-komponen utama (EOF1, EOF2)
- Detrending linear sebelum EOF
- Asosiasi EOF dengan fase ENSO (Niño3.4)

**Spesifikasi Domain:**
- **WMC (West Maritime Continent)**: 92.5-120.0°E, -12.5-12.5°N
- **EMC (East Maritime Continent)**: 120.0-152.5°E, -12.5-12.5°N

### 3. **Non-Stationarity Analysis** - Analisis Perubahan Temporal
Investigasi bagaimana kekuatan hubungan hujan-ENSO berubah seiring waktu:
- Pembagian data ke periode multiple decade
- Perhitungan korelasi per-period
- Pengujian signifikansi perubahan temporal
- Identifikasi faktor yang menyebabkan variabilitas temporal

### 4. **Lagged & Running Correlation** - Analisis Korelasi Tertinggal
- **Lag Analysis**: Mengukur keterlambatan optimal antara ENSO dan respons hujan
- **Running Correlation**: Jendela bergerak untuk analisis temporal perubahan korelasi
- Membantu memahami lead-lag relationship fisik

### 5. **ITCZ Analysis** - Analisis Zona Konvergensi
Menganalisis perilaku Intertropical Convergence Zone (ITCZ):
- Posisi dan intensitas ITCZ seasonal
- Sensitivitas ITCZ terhadap fase ENSO
- Pergeseran ITCZ di Maritime Continent
- Analisis menggunakan data GPCP (Global Precipitation Climatology Project)

### 6. **Circulation Analysis (850 hPa)** - Analisis Sirkulasi Atmosfer
Analisis medan sirkulasi di level tekanan 850 hPa:
- Pola angin pada musim DJF
- Anomali vorticity dan streamfunction
- Hubungan antara sirkulasi 850 hPa dan curah hujan lokal
- Moisture flux divergence dan convergence

## 🗂️ Data yang Digunakan

### Dataset Utama:
1. **MSWEP (Multi-Source Weighted-Ensemble Precipitation)**
   - File: `mswep_monthly_combined.nc`
   - Resolusi: 0.1° × 0.1°
   - Periode: 1980-2020
   - Variabel: Curah hujan bulanan

2. **ERA5 (European Reanalysis 5)**
   - Resolusi: 0.25° × 0.25°
   - Periode: 1980-2020
   - Variabel: Udara (U, V), kelembaban, tekanan
   - Level: Multiple levels termasuk 850 hPa

3. **Niño3.4 Index**
   - File: `nino34.anom.csv`
   - Sumber: NOAA/CPC
   - Data: Anomali SST di region Niño3.4 (5°N-5°S, 120°-170°W)

4. **GPCP (Global Precipitation Climatology Project)**
   - Untuk validasi dan ITCZ analysis
   - Resolusi: 2.5° × 2.5°

### Lokasi Data:
```
notebooks/
├── comprehensive_analysis/
├── divided_correlation/
├── eof_analysis/
│   └── data dari ../
├── non_stationarity/
├── lagged_correlation/
├── running_correlation/
├── itcz_analysis/
├── circulation850hpa/
├── cluster_enso/
└── (semua folder analisis)
```

**Catatan**: Notebook menggunakan relative paths untuk mengakses data, jadi pastikan struktur folder tetap konsisten.

## 🔧 Persyaratan Lingkungan

### Dependensi Python:
```
numpy
pandas
xarray
netCDF4
scipy
scikit-learn (untuk PCA/EOF)
matplotlib
seaborn
nbformat (untuk generate notebook)
python-dotenv (untuk konfigurasi path)
```

### Setup Environment:
```bash
# Menggunakan conda (rekomendasi)
conda create -n climate-analysis python=3.9
conda activate climate-analysis
conda install numpy pandas xarray scipy scikit-learn matplotlib seaborn

# Atau menggunakan pip
pip install numpy pandas xarray scipy scikit-learn matplotlib seaborn python-dotenv
```

### Software Tambahan:
- **Jupyter Notebook/JupyterLab**: Untuk menjalankan notebook interaktif
- **Python 3.8+**: Interpreter Python
- **Git**: Untuk version control (opsional tapi recommended)

## 🚀 Cara Menggunakan

### 1. Setup Data
Pastikan semua data input tersedia di lokasi yang ditentukan. Update konfigurasi sesuai dengan struktur folder lokal Anda:
```bash
# Verifikasi keberadaan data
ls -la ./external/ClimateData/
ls -la ../ClimateData/

# Atau gunakan config module
python -c "from data_processing.config import RAINFALL_PATH; print(RAINFALL_PATH)"
```

### 2. Menjalankan Notebook Interaktif
```bash
cd data_processing/notebooks
jupyter notebook comprehensive_analysis/rainfall_analysis_v3.ipynb
```

### 3. Generate Notebook dari Script Python
Beberapa notebook di-generate dari script Python:
```bash
# EOF Analysis
cd data_processing/notebooks/eof_analysis
python build_mswep_eof_mc_notebook.py

# Correlation Analysis
cd data_processing/notebooks/comprehensive_analysis/scripts
python build_correlation_global.py
python build_correlation_mc_v3.py
```

### 4. Menjalankan Analisis Running Correlation
```bash
cd data_processing/notebooks/running_correlation
python djf_runningcorr_domainjson_layoutAC.py
```

### 5. Mengerjakan Workflow Divided Correlation
```bash
cd data_processing/notebooks/divided_correlation
jupyter notebook dcorr_v6_workflow.ipynb
```

## 📈 Keluaran dan Hasil

### File Output:

#### 1. **Notebook Jupyter (.ipynb)**
- Dokumentasi lengkap dengan visualisasi
- Output interaktif dan reproducible
- Markdown explanations untuk setiap analysis step

#### 2. **Data NetCDF (.nc)**
Struktur tipikal output NetCDF:
```
Dimensions: (lat: 720, lon: 3600, time: ...)
Variables:
  - correlation: Koefisien korelasi Pearson
  - pvalue: P-value statistik
  - significant: Mask signifikansi (α=0.05)
  - rainfall: Rata-rata DJF
  - nino34: Indeks Niño3.4 standarisasi
Coordinates:
  - lat, lon, time
Attributes:
  - metadata tentang dataset
```

#### 3. **Visualisasi PNG**
- Peta korelasi spasial
- Time-series plot
- Diagram box plot per domain
- Klimatologi ITCZ
- Pola angin dan streamline

#### 4. **Tabel CSV/Excel**
- Statistik ringkasan
- Nilai korelasi puncak per region
- Tabel periode non-stationarity

### Contoh Hasil Analisis:
- **Korelasi Global DJF**: Menunjukkan region dengan korelasi kuat dengan Niño3.4
- **EOF Mode**: Mendeteksi variabilitas utama pola curah hujan
- **Temporal Evolution**: Perubahan korelasi dari periode 1980-2005 ke 2005-2020
- **ITCZ Sensitivity**: Pergeseran ITCZ di Maritime Continent saat El Niño/La Niña

## 📚 Referensi

### Dataset Referensi:
1. Beck, H. E., et al. (2019). "MSWEP V2 Global 3-Hourly 0.1° Precipitation: Methodology and Quantitative Assessment"
2. Hersbach, H., et al. (2020). "The ERA5 global reanalysis", Quarterly Journal of the Royal Meteorological Society
3. NOAA/Climate Prediction Center. "Cold & Warm Episodes by Season"

### Metode Analisis:
1. **EOF/PCA**: Jolliffe, I. T. (2002). "Principal Component Analysis" (2nd ed.)
2. **Correlation Analysis**: Pearson correlation dengan significance testing (t-test)
3. **Non-Stationarity**: Moving window correlation, change-point detection
4. **Spatial Analysis**: Kriging dan interpolation untuk gapless field

### Maritime Continent References:
1. Aldrian, E., & Dwi Susanto, R. (2003). "Identification of three dominant rainfall regions within Indonesia and their relationship to sea surface temperature"
2. Qian, J.-H. (2008). "Why does rainfall in the East Indian Ocean reverse between the two phases of El Niño?"
3. Juneng, L., & Tangang, F. T. (2005). "Evolution of ENSO-related rainfall anomalies in Peninsular Malaysia and Borneo"

## 👤 Author
**Nama**: Rizma Prawira  
**NIM**: 12822029  
**Institusi**: [Institut/Universitas]  
**Tahun**: 2025-2026

## 📝 Lisensi
Proyek penelitian ini tersedia untuk keperluan akademik dan penelitian.

## 📧 Kontak & Dukungan
Untuk pertanyaan atau diskusi tentang proyek ini, silakan hubungi penulis melalui institusi pendidikan.

---

**Last Updated**: April 2026

## 🔍 Tips untuk Navigasi dan Penggunaan

### Urutan Analisis yang Disarankan:
1. Mulai dengan `comprehensive_analysis/rainfall_analysis_v3.ipynb` untuk memahami data
2. Lanjut ke `comprehensive_analysis/wind_analysis_v3.ipynb` untuk dinamika
3. Explore `eof_analysis/` untuk pola utama variabilitas
4. Gunakan `divided_correlation/` untuk analisis periode-spesifik
5. Periksa `non_stationarity/` untuk temporal evolution
6. Lihat `itcz_analysis/` untuk mekanisme regional

## Tips Teknis:
- **Memory-intensive**: Beberapa notebook besar memerlukan 8GB+ RAM
- **Reproducibility**: Gunakan seed yang konsisten untuk random operations
- **Performance**: Loading MSWEP data full resolution dapat lambat; pertimbangkan subsetting
- **Version Control**: Simpan output notebook/figures di `git` untuk tracking perubahan

### Debugging Umum:
- **Import Error**: Pastikan semua packages terinstall dengan `conda list`
- **Data Not Found**: Periksa struktur folder dan relative paths di notebook
- **Memory Error**: Kurangi resolusi atau gunakan dask untuk lazy loading
- **Slow Computation**: Subset temporal/spatial domain untuk testing

---

## ℹ️ Dokumentasi Tambahan

**Notebook Output Management**:
- nbstripout otomatis menghapus outputs sebelum push ke GitHub
- Local notebooks Anda tetap memiliki outputs

---

## 🔐 Notebook Output Management

Repository ini menggunakan **nbstripout** untuk keamanan dan kebersihan:

### Bagaimana Cara Kerjanya?

- **Lokal (Komputer Anda)**: Notebook dengan outputs (untuk referensi)
- **GitHub**: Notebook tanpa outputs (privacy & clean)
- **Anda**: Tidak perlu berbuat apa-apa! Semua otomatis

### Mengapa Penting?

Notebook outputs bisa berisi path sensitif seperti:
```
/Users/yourname/miniforge3/envs/climate_data/lib/python3.13/site-packages/shapely/creation.py:730
```

Dengan nbstripout:
- ✅ Outputs dihapus sebelum push ke GitHub
- ✅ Local files tetap punya outputs
- ✅ Tidak ada path sensitif di GitHub

### Setup Untuk Collaborators

```bash
# Install nbstripout
pip install nbstripout

# Configure git filter
git config filter.nbstripout.clean 'nbstripout'
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

Atau baca [`NBSTRIPOUT_GUIDE.md`](../NBSTRIPOUT_GUIDE.md) untuk detail lengkap.
