## 勇往職前 - AI求職招聘媒合助理-聊天機器人(團隊合作專案)

專案組長: 張維芸 <br>
負責部分: 專案管理、功能設計、資料清洗、特徵工程、推薦模型建立、模型部屬 <br>
完成日期: 2020年11月 <br>
使用技術: 機械學習(Scikit-Learn : Bayes classifier, Grid Search)訓練分類模型、NLP自然語言處理(BOW, Word2Vec) <br>

主要製作功能 : <br>
    1. 找職缺: 求職者輸入自身技能與履歷，推薦適合的104人力銀行職缺。(此repository展示)<br>
    2. 找履歷: HR輸入人才需求，推薦適合的CakeResume履歷表(製作方式類似找職缺)

*注意: 此作品使用繁體中文做為文本訓練，因此用戶輸入繁體中文履歷效果比較好。 <br>

功能展示影片: (製作中) <br>
功能展示PPT: [勇往職前PPT](https://docs.google.com/presentation/d/1bxf6blvl5OK7HTN-d9_5F_2EXNDgBhjVYbUTWAQVWjg/edit?usp=sharing "勇往職前PPT")(製作中) <br>
功能展示網址: [找職缺網址](http://for-workers.herokuapp.com/html2 "找職缺網址") <br>

--------------------------------------------------------------
####  完整的(找職缺)功能執行代碼為: "Job_recommendation.py"
> 求職者輸入自身技能與履歷，推薦10筆適合的104人力銀行職缺。

以下為功能流程:

> 1. 使用者輸入CV，以及工作條件

> 2. 清洗CV，轉換成BOW， 丟入sklearn訓練模型，顯示預測分類結果。

> 3. 利用預測分類結果，與使用者輸入工作條件，從mongoDB撈取職缺

> 4. 將CV轉換成Word2Vec可用BOW

> 5. 計算CV與職缺的相似度，(使用Word2Vec模型)推薦相似度最近的10筆工作

--------------------------------------------------------------
此repository的Jupyter Notebook檔案顯示(找職缺)功能的製作詳細過程，流程如下:

#### 步驟1 - 職缺名稱清洗
    代碼展示: "01_Data_Cleaning_Job_Title.ipynb"

#### 步驟2 - 工作內容清洗
    代碼展示: "02_Data_Cleaning_Job_Description.ipynb"

#### 步驟3 - 建立文本分類模型
    代碼展示: "03_Build_text_classification_model.ipynb"

#### 步驟4 - 建立推薦系統 by cosine similarity
    代碼展示: "04_Build_recommendation_by_Similarity.ipynb"

#### 步驟6 - 建立推薦系統 by Word2Vec
    代碼展示: "05_Build_recommendation_by_Word2Vec.ipynb"


--------------------------------------------------------------

其他文件說明:

1. jieba_data: 儲存結巴分詞文件與stopwords文件。
2. model: 儲存訓練好的機械學習模型

