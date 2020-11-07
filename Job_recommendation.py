from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import pickle
import jieba
import re
import pymongo
import pandas as pd
from secrets import HOST


def string_clean(CV_data):
    # define data cleaning function
    '''
    input: string (original CV description)
    output: string (clean CV description)
    '''
    job_desc = CV_data.split('\n')  # 根據換行符號轉乘 List格式
    job_words = ''

    for words in job_desc:
        words = words.replace('\t', ' ').replace('\r', ' ')
        words = re.sub(r'[^\w\s]', ' ', words)  # remove all punctuations
        words = re.sub(r'\d+', ' ', words)  # remove all numbers
        words = words.strip()  # remove white space
        words += ' '
        job_words += words

    return (job_words)


def isEnglish(s):  # 檢查字元是否為英文
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def jieba_cut(data, stop_words):
    # 使用結巴斷詞
    seg_result = jieba.cut(data, cut_all=False)

    # 篩選斷詞，去掉單一中文字
    lst_seg = []

    for i in list(seg_result):
        i = i.strip()
        if len(i) < 1:  # 排除空值
            continue
        elif i in stop_words:  # 排除stopwords
            continue
        elif isEnglish(i) == False and len(i) == 1:  # 排除單一中文字
            continue
        else:
            lst_seg.append(i)

    return lst_seg


def cv_category_predict(cv_data):
    vectorizer = pickle.load(open("model/vectorizer.pickel", "rb"))
    transform_content = vectorizer.transform(cv_data)  # CountVectorizer transform
    X_test = transform_content.toarray()
    # vectorizer.inverse_transform(X_test)

    model = load('model/MultinomialNB.joblib')
    y_pred = model.predict(X_test)  # MultinomialNB transform

    le = pickle.load(open("model/le.pickel", "rb"))
    y_pred_category = le.inverse_transform(y_pred)
    y_pred_category = y_pred_category[0]

    return y_pred_category


def mongo_select_jobs(category, area=None, workExp=None, edu=None):
    global mycol
    client = pymongo.MongoClient(HOST)

    db = client['final_104']
    mycol = db['final_data']

    filter_dict = {'jobCat_main': category}  # create filter for mongoDB

    if area != None:
        filter_dict['Work_area_clean'] = area  # filter data by area

    data = mycol.find(filter_dict, {'_id': 0})  # get data from mongoDB
    output = [i for i in data]

    # turn output into dataframe
    df = pd.DataFrame(output)

    # filter data by work experience
    if workExp is not None:
        lst_workExp = list(i for i in range(0, workExp + 1))
        df = df[df['workExp_clean'].astype(int).isin(lst_workExp)]

        # filter data by education status
    if edu is not None:
        df = df[df['Edu_clean'].str.contains("大學")]

    filter_result = df.values.tolist()

    # 將資料儲存成flask可用的dictionary格式

    dict_data = {}

    for i in filter_result:
        jobURL, Job_Name, Company, Job_Description, concate_jieba = i[0], i[1], i[2], i[8], i[9]
        dict_data[jobURL] = {
            'Job_Name': Job_Name,
            'Company': Company,
            'Job_Description': Job_Description,
            'concate_jieba': concate_jieba}

    return dict_data


def turn_content_BOW(content):  # 將結巴後的斷詞，利用CountVectorizer轉換減少字詞數量
    X_content = [ " ".join(content)]
    vectorizer = pickle.load(open("model/vectorizer.pickel", "rb"))
    transform_content = vectorizer.transform(X_content)
    X_content_array = transform_content.toarray()
    return X_content_array


def compute_similarity(cv_BOW, job_id, job_jieba_data):
    job_BOW = turn_content_BOW(job_jieba_data)
    job_prob = cosine_similarity(cv_BOW, job_BOW)
    job_prob = round(job_prob.tolist()[0][0], 6)

    return job_prob, job_id


def show_recommendation_result(cv_clean, jobs_query):
    cv_BOW = turn_content_BOW(cv_clean)

    dict_prob_id = {}

    for i, j in jobs_query.items():
        job_jieba_data = j['concate_jieba']
        similarity = compute_similarity(cv_BOW, i, job_jieba_data)
        dict_prob_id[similarity[0]] = similarity[1]
        result = [(k, dict_prob_id[k]) for k in sorted(dict_prob_id.keys(), reverse=True)[0:10]]

    list_ten_result = []

    for i, j in result:
        data = jobs_query[j]
        lst_result = [i, data['Job_Name'], data['Company'], data['Job_Description'], j]
        list_ten_result.append(lst_result)

    return (list_ten_result)


def main():
    # 1. 使用者輸入CV，以及工作條件 ---------------------------------------------------------------

    # input_cv = str(input('請輸入中文CV: '))
    # input_work_area = str(input('請輸入理想工作縣市: '))
    # input_work_exp = int(input('請輸入工作年分(數字): '))
    # input_edu = str(input('請輸入最高教育: '))

    input_cv = '''最近工作\n\n公司：\tXX醫院\t行業：\t醫療/護理/衛生\n職位：\t醫生\n最高學歷\t\n
    最高學歷\n\n學校：\t中醫藥大學\n學歷：\t本科\t專業：\t\n醫藥學\n工作經驗\t\n工作經驗\n\n公司：\t\n
    XX醫院\n2012/7–2017/7\n職位：\t醫生\n行業：\t醫療/護理/衛生\n部門：\t醫藥部\n工作內容：\n1.
    了解各種儀器設備的使用方法。2.參與手術工作，鍛煉手術操作能力。3.熟悉實際操作中所出現的問題並通過各種
    方法避免和克服。4.勤學好問，大膽展示自我，學會了要禮貌待人，要踏實幹事，要提高個人綜合素質。\n教育經歷\t\n
    教育經歷\n\n學校：\t\n中醫藥大學\n2007/9–2011/6\n專業：\t醫藥學\t本科\t\n自我評價\t\n自我評價\n\n在校期
    間，學習了解剖學、生物化學、生理學、病理學、精神學等等課程，在校成績優異，擔任學生幹部的職務，曾多次獲得過
    學校獎學金，平時積極主動參加校內活動，曾負責在專業內組織過醫學演講。學習態度端正，能夠主動學習，認真對待每
    一次學習的機會，我希望在學習理論知識的同時，能夠增強自己的實踐經驗，我 相信沒有做不到，只有不想做。\n求職
    意向\t\n求職意向\n\n到崗時間：\t一個月之內\n工作性質：\t全職\n希望行業：\t醫療/護理/衛生\n目標地點：\t
    北京\n期望月薪：\t面議/月\n目標職能：\t醫生\n語言能力\t\n語言能力\n\n英語：\t\n良好\n\n聽說：\t\n良好
    \n\n讀寫：\t\n良好\n\n證書\t\n證書\n\n大學英語四級'''

    input_work_area = '台北市'
    input_work_exp = 6
    input_edu = '大學'

    # 2. 清洗CV，轉換成BOW， 丟入訓練模型，顯示預測分類結果。  ------------------------------------

    jieba.load_userdict('jieba_data/Jobcontent_dict.txt')  # 指定辭典檔

    with open(file='jieba_data/Jobcontent_stopwords.txt', mode='r', encoding="UTF-8") as file:
        # 排除字元表單 stopword, 開啟 'Jobcontent_stopwords.txt'檔案
        stop_words = file.read().split('\n')
        stop_words = [i.strip() for i in stop_words]

    input_cv_clean = string_clean(input_cv)
    input_cv_clean = jieba_cut(input_cv_clean, stop_words)
    input_cv_clean = [' '.join(input_cv_clean)]

    input_cv_category = cv_category_predict(input_cv_clean)

    # 3. 利用預測分類結果，與使用者輸入工作條件，從mongoDB撈取職缺  --------------------------

    jobs_query = mongo_select_jobs(input_cv_category, input_work_area, input_work_exp, input_edu)

    # 4. 將職缺轉換成BOW，計算CV與職缺的相似度，推薦相似度最近的10筆工作  ------------------------------------

    final_result = show_recommendation_result(input_cv_clean, jobs_query)
    print(final_result)


if __name__ == '__main__':
    main()