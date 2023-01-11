#%%
import numpy as np
import matplotlib.pyplot as plt
import reedSolomon
import time
import tqdm

#%% Reed Solomon Kodlayıcı/Kodcozucu Parametreleri
# GF(q) icin q degeri
q = 256

# GF(q) icin primitif polinom, veri tipi "int64" olmalı
# p(X) = X^8 + X^5 + X^3 + X + 1
prim_poly = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1], dtype = np.int64)

# GF(q) icin primitif eleman (2 secilmesinde bir sakinca yok)
prim_elem = 2

# Her Reed Solomon simgesi cinsinden giris cikis uzunluklari
n = 255 # cikis RS simge dizisi 
k = 128 # giris RS simge dizisi

# Reed solomon uretec polinomu birincil carpan usteli
b = 1
# "b" degerinin kullanimina ornek: Kodlayici asagidaki uretec polinomunu olusturacak
# g(X) = (X+alpha^(b))*(X + alpha^(b+1))*...*(X + alpha^(b+2*t))
# Burada "alpha" ile primitiv eleman ve "t" ile duzeltilebilir RS sembol hatasi sayisi gosteriliyor.

# Reed Solomon kodlayıcı ve kodcozucu nesnelerini olustur
rs_encoder = reedSolomon.ReedSolomonEncoder(q, n, k, prim_poly, prim_elem = prim_elem, b = b)
rs_decoder = reedSolomon.ReedSolomonDecoder(q, n, k, prim_poly, prim_elem = prim_elem, b = b)

#%% SIMULASYON

# Tekrar sayisi (kac adet RS kodlanmis kodsozcugunun gonderilecegi)
n_MC = int(1e3)

# Simulasyonun hangi SNR [dB] degerleri icin yapilacagi
snr_db = np.arange(0,10,0.2)

#Bit ve frame hata orani sayaci
ber    = 0.0*snr_db
fer    = 0.0*snr_db

# Bir kodsozcugu kac adet bilgi bitinden olusmakta
n_bit_for_u = k*rs_encoder.num_bits_for_RS_symbol

#Simulasyon baslangici
for ss in range(len(snr_db)):
    # SNR degeri lineer bicimde
    snr = 10.0**(snr_db[ss]/10.0)

    # Simulasyonunun durduruldugunu belirten bayrak
    break_flag = False

    for ii in tqdm.trange(n_MC):
        
        # Rastgele bitler uret
        u_bit = np.random.randint(0, high = 2, size = n_bit_for_u, dtype = np.int64)
        
        # Reed Solomon kodlama yap
        v     = rs_encoder.encode_bit_to_bit(u_bit, order = 'MSB_first')

        # BPSK module et
        x = 2.0*np.single(v) - 1.0
        
        # Gurultu ekle
        noise = 1/np.sqrt(2*snr*k/n)*np.random.randn(len(x))
        y = x + noise
        
        # BPSK demodule et
        r = np.array(y>0, dtype=np.int64)

        # RS kod coz
        v_hat    = rs_decoder.decode_bit_to_bit(r, order= 'MSB_first')
        
        # Sistematik kodlama sabebiyle bilgi bit dizisini cek
        u_bit_hat    = v_hat[:n_bit_for_u] 
        #BER ve FER say
        ber_iter = np.sum(u_bit_hat != u_bit)
        ber[ss] += ber_iter
        fer[ss] += 1*(ber_iter>0)
        
        #Simulasyonu durdur eger yeterli sayida bit hatasi gozlemlendiyse
        if fer[ss]> 100:
            #BER ve FER degerlerini normalize et
            ber[ss] /= (ii*n_bit_for_u)
            fer[ss] /= ii

            #Simulasyonun durdurulduguna dair bayragi kaldir
            break_flag = True
            break
    # Eger bayrak kalktıysa BER ve FER degerlerini normalize et
    if break_flag == False:
        ber[ss] /= (n_MC*n_bit_for_u)
        fer[ss] /= n_MC

# %%
# Sonuclari cizdir
plt.figure(1)
plt.semilogy(snr_db,ber,'--*',label = 'BER')
plt.semilogy(snr_db,fer,'-*', label = 'FER')
plt.xlabel('Ec/No')
plt.ylabel('Error Rate')
plt.grid(which='both',ls='--',alpha = 0.5)
plt.legend()
# %%

# %%
