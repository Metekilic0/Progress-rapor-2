KOD OLARAK YÜKLEDİĞİM MODELLER;
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

Sonuç;
Projemde 2. Yöntem olan CNN modeli eğitme konusunda önemli derecede ilerleme kaydettiğinimi düşünüyorum. İlk 
model %81 başarı yüzdesinden başlarken en gelişmiş modelim %98 bir başarı yüzdesi yakaladı. Bu modelde 
olabilecek birçok şeyi test ettim. Bunları biraz detaylandırmak gerekirse son modelime ulaşırken bu modelde Batch 
Normalization, Dynamic Learning Rate, 5 katmanlı Conv2D, Sınıf ağırlıklandırma, EarlyStopping ve Checkpoint, 
sınıflandırıcı olarakGlobalAveragePooling2D, Dropout katmanı kullandım. Optimizer için Adam’ı kullandım.  Bu 
projede beklediğim beklemediğim tüm sonuçları modellerin altında detaylıca belirttim. Transfer learning modeli olan 
mobilenetv2 kullandım ve çok başarılı sonuçlar elde ettim. 
Burada kısaca belirtmek gerekirse Data augmentation konusunda beklediğim sonuçları alamadım ve modelde 
kullanmadım. Buna benzer şekilde küçük veri setinde yüksek çözünürlük, ek bir dropout katmanı, Dense ve flatten 
katman sayısı düşürme gibi yöntemler beklediğim sonuçları vermediği için modelde kullanmadım. En sonda ise 
büyük veri setimi test ettim ve %98 doğruluk yüzdesine ulaştım. 