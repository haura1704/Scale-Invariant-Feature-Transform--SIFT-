# Image Matching Menggunakan SIFT dan RANSAC

## Deskripsi
Program ini mengimplementasikan metode Image Matching menggunakan algoritma SIFT untuk deteksi dan deskripsi fitur, serta RANSAC untuk menyaring pencocokan yang tidak valid. Sistem digunakan untuk mencocokkan dua citra yang memiliki bagian objek yang sama.

## Kebutuhan
- Python 3.x
- opencv-python
- opencv-contrib-python
- numpy
- matplotlib

Instalasi:
pip install opencv-python opencv-contrib-python numpy matplotlib

## Cara Menjalankan
1. Pastikan folder images berisi:
   - taman.jpg
   - airmancurtaman.jpg
2. Jalankan:
   python image_matching.py

## Output
Program menampilkan:
- Dua citra input
- Hasil pencocokan fitur
- Jumlah good matches
- Jumlah inliers setelah RANSAC

## Penulis
Aisha Haura
Teknik Informatika
