#!/usr/bin/env python
# coding: utf-8

# In[561]:


import pandas as pd            #pandas kütüphanesini pd komutuyla kullanıyoruz.
train=pd.read_csv("train.csv") #train ve test csv verilerimizi pandas ile getiriyoruz.
test=pd.read_csv("test.csv")


# In[562]:


train.head() #ilk 5 veriyi getirerek veri kontrolü yapıldı.


# In[563]:


test.head() #ilk 5 veriyi getirerek veri kontrolü yapıldı.


# In[564]:


train.shape # boyut ve o boyuttaki eleaman sayısı getirildi.(891 veride 12 tane ayrı sütun olduğu görüldü.)


# In[565]:


test.shape #(418 veride 11 tane ayrı sütun olduğu görüldü.)


# In[566]:


train.info() #(Her sütundaki, ilgili bilgiler görüldü. )


# In[567]:


test.info() #(Her sütundaki, ilgili bilgiler görüldü. )


# In[568]:


#Birçok satır için yaş değerinin eksik olduğu görülmekte.
#891 satırdan yaş değeri yalnızca 714 satırda var.
#kabin değerleri çoğu sırada eksik,891 sıradan sadece 204 tanesinde kabin değeri var.


# In[569]:


train.isnull().sum() #train.csv veri setindeki eksik verilerin sayıları getirildi.


# In[570]:


test.isnull().sum() #test.csv veri setindeki eksik verilerin sayıları getirildi.


# In[571]:


import matplotlib.pyplot as plt #matplotlip ile veri görselleştirmek için import edildi.
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #istatiksel grafikler yapmak için seaborn import edildi.
sns.set() #plot için seaborn varsayılan olarak ayarlama.


# In[572]:


def grafik(ozellik):  #grafik fonksiyonu tanımlandı.ozellik classı oluşturuldu.
    yasayanlar=train[train["Yasayanlar"]==1][ozellik].value_counts() #train.csv içindeki yasayan insanlar sayıldı.
    olenler=train[train["Yasayanlar"]==0][ozellik].value_counts() #train.csv içindeki ölen insanlar sayıldı.
    df=pd.DataFrame([yasayanlar,olenler]) #DataFrame yapısına aldık.
    df.index=["yasayanlar","olenler"] # grafikteki baslıklarımız yaşayanlar ve ölenler şeklindedir.
    df.plot(kind="bar",stacked=False,figsize=(5,5))  #Tür grafik olarak seçip , grafik boyutlarımızı ayarladık.


# In[573]:


grafik("Cinsiyet") #fonksiyonumuzdaki cinsiyete göre yasayan-ölüm grafiğini getirdik.


# In[574]:


#Grafikte kadınların erkeklere göre daha fazla hayatta kaldığını doğruluyor.


# In[575]:


grafik("Sinif") #sınıflara göre ölüm olanlarını getirdik.


# In[576]:


#Grafikte 1.sınıf insanların daha fazla hayatta kaldığı görülmektedir.
#3.sınıf insanlarınsa daha fazla öldüğü görülmektedir.


# In[577]:


#Veri setine göre kullanabileceğim 2 tane özellik olduğunu düşünüyorum;
    #1.si cinsiyet , 2.si sınıf
    #Veri setine göre tahminlememi cinsiyetlerine ve sınıflarına göre yapacağım.
    #Kullanacağım regresyon çeşidi ise Logistic Regresyondur.
    #Sebebi; bir sonucu bir veya birden fazla bağımsız değişken bulunan bir veri kümesini
    #analiz edebilmek için yeterli olacaktır.
#Bu şekilde hangi yolcularının kurtulduğunu tahmin eden bir model geliştireceğim.


# In[578]:


train["Ucret"].hist(bins=10,figsize=(10,7))


# In[579]:


train["Kardes/es"].hist(bins=10 , figsize=(10,5))
#grafikte çoğu kişinin ücretlerinin düşük olduğunu görüyoruz.
#kardeş/eş grafiğindeyse çoğu kişinin gemiye yalnız bindiğini görüyoruz.


# In[580]:


plt.figure(figsize=(10,5))
sns.boxplot(x="Sinif",y="Yas", data=train);


# In[581]:


#Daha çok 30-50 yaş arası insanların 1.sınıf insanlar olduğunu görüyoruz.
#2.sınıf insanlarınsa 25-35 yaş arası insanlar olduğunu görebiliyoruz.
#3.sınıf insanlarınsa 20-30 yaş arası insanlar olduğunu görebiliyoruz.


# In[582]:


train.head()


# In[583]:


train["Yas"] #NaN değerleri ortalama bir değerle doldurmam gerekiyor.


# In[584]:


def impute_yas(col):
    Yas=col[0]
    Sinif=col[1]
    
    if pd.isnull(Yas):
        if Sinif==1:
            return 40
        elif Sinif==2:
            return 30
        else:
            return 25
    else:
        return Yas


# In[585]:


train["Yas"]=train[["Yas","Sinif"]].apply(impute_yas,axis=1)


# In[586]:


train["Yas"] #NaN değerlerimiz ortalama değerlerle doldu.


# In[587]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# In[588]:


train.drop("Kabin",axis=1,inplace=True)


# In[589]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# In[590]:


#Grafikteki kabin ortadan kalktı.


# In[591]:


train.dropna(inplace=True) #tüm kayıp verileri ortadan kalkdırdık.


# In[592]:


Cinsiyet=pd.get_dummies(train["Cinsiyet"],drop_first=True) #Cinsiyet string tipinde olduğu için float tipine dönüşüm sağladım.


# In[593]:


Cinsiyet.head()


# In[594]:


Binis_Yeri=pd.get_dummies(train["Binis_Yeri"],drop_first=True)#(dummy veriable)


# In[595]:


Binis_Yeri.head()


# In[596]:


train=pd.concat([train,Cinsiyet,Binis_Yeri],axis=1) #Dönüşüm sağladığım verilerin tabloyla birleşimini sağladım.


# In[597]:


train


# In[598]:


train.drop(["Yolcu_kimligi","İsim","Cinsiyet","Bilet","Ucret","Binis_Yeri"],axis=1,inplace=True)


# In[599]:


train


# In[600]:


#Datamız artık regresyon datası haline getirildi.


# In[601]:


X=train.drop("Yasayanlar",axis=1)
y=train["Yasayanlar"]


# In[602]:


from sklearn.model_selection import train_test_split


# In[603]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[604]:


from sklearn.linear_model import LogisticRegression


# In[605]:


logmodel=LogisticRegression()


# In[606]:


logmodel.fit(X_train, y_train)


# In[607]:


Tahminler=logmodel.predict(X_test)


# In[608]:


from sklearn.metrics import classification_report


# In[609]:


print(classification_report(y_test,Tahminler))


# In[610]:


print("Modelin Doğruluğu=%84'tür.")


# In[614]:


Tahminler


# In[615]:


print("Saygılar.")

