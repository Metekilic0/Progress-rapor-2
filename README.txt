KOD OLARAK YÜKLEDİĞİM MODELLER;
V12 MODEL -- SINIF AĞIRLIKLANDIRMA YAPTIĞIM MODEL VE %94.85 DOĞRULUK ORANI
V20 MODEL --- 5. KATMAN EKLENMİŞ MODELİN EN İYİ HALİ %95,94 DOĞRULUK ORANI 
MOBİLENETV2 --- GOOGLE'IN MOBİLENETV2 MODELİ İLE TRANSFER LEARNİNG %98,59 DOĞRULUK ORANI
V21 MODEL --- BÜYÜK VERİ SETİNDE V20 MODEL İLE AYNI SADECE SINIFLARI HELİKOPTER VE DRONE DA DAHİL OLACAK ŞEKİLDE GÜNCELLEDİM. 

VERİ SETLERİM;
final_dataset1 kuş veri setim sadece kuş ve drone var.V21 MODEL HARİÇ TÜM MODELLERİM BU VERİ SETİNİ KULLANDI.
dataset4 büyük veri setim tüm sınıfları içeriyor. V21 Modelde kullandım %98,26 doğruluk yüzdesi elde ettim.

DENEME VERİ SETLERİM;
dataset1_küçük Küçük veri setim için 100-20-20 dağılmış örnek veri
dataset4_küçük Büyük veri setim için 100-20-20 dağılmış örnek veri

MODELLERİM;
best_4class_model.keras --- en son kullandığım büyük veri setimdeki model yani V21
best_model.keras --- V20 yani küçük veri setiminin en gelişmiş hali
best_tl_model.keras --- MobileNetV2 kullanılmış transfer learning modelim 

