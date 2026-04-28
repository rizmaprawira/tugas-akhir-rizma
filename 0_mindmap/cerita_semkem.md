# Cerita Semkem

## Latar Belakang
Fenomena ENSO diketahui memengaruhi curah hujan di Indonesia melalui telekoneksi. Sejumlah studi sebelumnya juga menunjukkan bahwa pengaruh ENSO terhadap hujan di Indonesia tidak seragam pada musim DJF. Pertanyaan utamanya adalah apakah ketidakseragaman tersebut sudah ada sejak dulu, atau baru mulai menguat pada periode yang lebih baru.

Sebagai penghubung cerita, perubahan ini dapat dikaitkan dengan perubahan kondisi ENSO yang juga teramati dalam literatur, tetapi bagian tersebut tidak menjadi fokus analisis utama. Fokus penelitian ini adalah perubahan hubungan ENSO terhadap curah hujan di Indonesia, waktu mulai berubahnya, besar perubahannya, dan kemungkinan penyebab dinamisnya, terutama pada sirkulasi level bawah di 850 hPa.

## Pertanyaan Penelitian
1. Apakah ada perubahan hubungan ENSO terhadap curah hujan di Indonesia?
2. Jika ada, sejak kapan perubahan tersebut mulai terlihat?
3. Seberapa besar perbedaan hubungan ENSO-hujan pada dua periode tersebut?
4. Apa kemungkinan penyebab perubahan tersebut, terutama dari sisi sirkulasi level bawah pada 850 hPa?

## Data dan Pra-pemrosesan
Analisis utama menggunakan data curah hujan MSWEP bulanan resolusi 0.1 degree dan indeks Niño 3.4 bulanan. Untuk analisis dinamika atmosfer, digunakan data ERA5 bulanan, meliputi angin zonal dan meridional pada 850 hPa, streamfunction dan rotational wind pada 850 hPa, geopotential height 850 hPa, serta mean sea level pressure (MSLP). Untuk pendukung analisis hubungan hujan-ENSO, digunakan juga keluaran korelasi yang disimpan dalam bentuk cache NetCDF agar peta dapat ditampilkan kembali tanpa menghitung ulang.

Seluruh data bulanan kemudian disusun ke dalam musim DJF dengan konvensi yang konsisten, yaitu DJF1981 didefinisikan sebagai Desember 1980, Januari 1981, dan Februari 1981. Hanya musim DJF yang lengkap yang dipakai dalam analisis. Pada analisis korelasi utama, data curah hujan dan Niño 3.4 juga diubah menjadi anomali terhadap klimatologi 1991-2020, kemudian didetrend secara linear sebelum korelasi dihitung.

## Metode Analisis
### 1. Korelasi grid-level ENSO terhadap hujan
Hubungan ENSO terhadap curah hujan dihitung dengan korelasi Pearson pada setiap grid hujan terhadap indeks Niño 3.4 DJF. Analisis ini dilakukan untuk seluruh periode pengamatan, kemudian dibandingkan antarperiode untuk melihat apakah pola telekoneksi berubah dari waktu ke waktu.

### 2. Analisis split periode
Untuk menguji kestasioneran hubungan ENSO-hujan, periode DJF 1981-2025 dibagi ke beberapa kandidat split dengan panjang periode awal 10 sampai 35 tahun. Setiap split menghasilkan pasangan periode P1 dan P2. Perubahan dinilai dari besar perubahan korelasi grid, termasuk grid yang mengalami perubahan mutlak korelasi pada beberapa ambang, yaitu 0.4, 0.6, dan 0.8, serta dari rata-rata besar perubahan korelasi di wilayah darat, laut, dan seluruh grid.

Split yang paling menonjol menunjukkan perubahan terjadi pada batas 2006/2007, sehingga periode utama dibagi menjadi 1981-2006 dan 2007-2025. Dari perbandingan ini, periode awal memperlihatkan hubungan ENSO-hujan yang lebih seragam di seluruh Indonesia, sedangkan periode akhir mulai menunjukkan ketidakseragaman, terutama di Sumatra, Kalimantan, dan Sulawesi.

### 3. Korelasi lag curah hujan dan Niño 3.4
Sebagai pelengkap, dibuat analisis korelasi lag antara indeks curah hujan DJF dan Niño 3.4 rata-rata berjalan 3 bulan yang terpusat. Lag dihitung dari -12 sampai +12 bulan dengan konvensi DJF yang sama. Analisis ini dikerjakan untuk dua periode, yaitu 1981-2006 dan 2007-2025, lalu selisih kurva dihitung sebagai P2 - P1.

### 4. Korelasi lag spasial
Selain untuk kotak curah hujan, korelasi lag juga dihitung secara spasial untuk seluruh grid curah hujan DJF terhadap Niño 3.4 rata-rata berjalan 3 bulan. Hasilnya disimpan dalam cache NetCDF agar plot lag tertentu dapat ditampilkan ulang tanpa menghitung korelasi dari awal.

### 5. Peta korelasi lintas variabel
Untuk mendukung interpretasi fisik, peta korelasi juga ditampilkan untuk beberapa variabel atmosfer, yaitu curah hujan, angin zonal dan meridional, moisture flux convergence, streamfunction, velocity potential, MSLP, dan geopotential height 850 hPa. Peta-peta ini disiapkan untuk domain Indo-Pasifik dan Maritime Continent dengan menggunakan keluaran NetCDF korelasi yang sudah disimpan sebelumnya.

### 6. Analisis komposit
Untuk melihat perubahan respons atmosfer terhadap ENSO, dibuat komposit DJF untuk kondisi El Niño dan La Niña pada dua periode, yaitu 1981-2006 dan 2007-2025, lalu dibandingkan dengan periode penuh 1981-2025 sebagai baseline. Analisis komposit dilakukan pada angin 850 hPa, streamfunction dan rotational wind 850 hPa, geopotential height 850 hPa, serta MSLP. Pada komposit angin, kecepatan angin ditampilkan sebagai warna dan vektor angin ditumpangkan sebagai quiver.

## Hasil Sementara
Pada periode 1981-2006, hubungan ENSO dengan hujan DJF masih lebih seragam di seluruh Indonesia. Namun, pada periode 2007-2025, hubungan tersebut mulai tidak seragam, terutama di Sumatra, Kalimantan, dan Sulawesi. Di wilayah-wilayah tersebut, dampak ENSO justru cenderung berlawanan dari pola yang biasanya muncul di Indonesia saat El Niño.

## Interpretasi Sementara
Hasil analisis sementara menunjukkan adanya peningkatan angin dari selatan ke utara pada level bawah. Aliran ini menabrak Sumatra, Kalimantan, dan Sulawesi setelah membawa uap air melewati Laut Jawa, sehingga memicu konvergensi uap air di wilayah tersebut.

Hal yang masih belum terjawab adalah mengapa sirkulasi level bawah itu bisa berubah mulai sekitar tahun 2006. Bagian ini menjadi analisis lanjutan.
