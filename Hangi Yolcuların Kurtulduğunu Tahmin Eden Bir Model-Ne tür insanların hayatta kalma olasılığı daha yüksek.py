#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd            #pandas kütüphanesini pd komutuyla kullanıyoruz.
train=pd.read_csv("train.csv") #train ve test csv verilerimizi pandas ile getiriyoruz.
test=pd.read_csv("test.csv")


# In[32]:


train.head() #ilk 5 veriyi getirerek veri kontrolü yapıldı.


# In[33]:


test.head() #ilk 5 veriyi getirerek veri kontrolü yapıldı.


# In[34]:


train.shape # boyut ve o boyuttaki eleaman sayısı getirildi.(891 veride 12 tane ayrı sütun olduğu görüldü.)


# In[35]:


test.shape #(418 veride 11 tane ayrı sütun olduğu görüldü.)


# In[36]:


train.info() #(Her sütundaki, ilgili bilgiler görüldü. )


# In[37]:


test.info() #(Her sütundaki, ilgili bilgiler görüldü. )


# In[38]:


#Birçok satır için yaş değerinin eksik olduğu görülmekte.
#891 satırdan yaş değeri yalnızca 714 satırda var.
#kabin değerleri çoğu sırada eksik,891 sıradan sadece 204 tanesinde kabin değeri var.


# In[39]:


train.isnull().sum() #train.csv veri setindeki eksik verilerin sayıları getirildi.


# In[40]:


test.isnull().sum() #test.csv veri setindeki eksik verilerin sayıları getirildi.


# In[25]:


import matplotlib.pyplot as plt #matplotlip ile veri görselleştirmek için import edildi.
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #istatiksel grafikler yapmak için seaborn import edildi.
sns.set() #plot için seaborn varsayılan olarak ayarlama.


# In[26]:


def grafik(ozellik):  #grafik fonksiyonu tanımlandı.ozellik classı oluşturuldu.
    yasayanlar=train[train["Yasayanlar"]==1][ozellik].value_counts() #train.csv içindeki yasayan insanlar sayıldı.
    olenler=train[train["Yasayanlar"]==0][ozellik].value_counts() #train.csv içindeki ölen insanlar sayıldı.
    df=pd.DataFrame([yasayanlar,olenler]) #DataFrame yapısına aldık.
    df.index=["yasayanlar","olenler"] # grafikteki baslıklarımız yaşayanlar ve ölenler şeklindedir.
    df.plot(kind="bar",stacked=False,figsize=(5,5))  #Tür grafik olarak seçip , grafik boyutlarımızı ayarladık.


# In[27]:


grafik("Cinsiyet") #fonksiyonumuzdaki cinsiyete göre yasayan-ölüm grafiğini getirdik.


# In[28]:


#Grafikte kadınların erkeklere göre daha fazla hayatta kaldığını doğruluyor.


# In[29]:


grafik("Sinif") #sınıflara göre ölüm olanlarını getirdik.


# In[30]:


#Grafikte 1.sınıf insanların daha fazla hayatta kaldığı görülmektedir.
#3.sınıf insanlarınsa daha fazla öldüğü görülmektedir.


# In[31]:


grafik("Kardes/es") 


# In[32]:


#Grafiğe göre gemiye  yalnız binenler daha çok ölmüştür.


# In[33]:


grafik("Ebevyn/Cocuklar")


# In[34]:


#Grafiğe göre gemiye yalnız binenler daha çok ölmüştür.


# In[35]:


grafik("Binis_Yeri")


# In[36]:


#Grafiğe göre;
#Southamptondan binenler daha fazla ölmüştür.
#Cherbourgtan binenlerin çoğu hayatta kalmıştır.
#Queenstowndan binenler daha fazla ölmüştür.


# In[ ]:





# In[ ]:




