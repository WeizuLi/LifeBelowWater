
from tqdm import tqdm
import time
import requests
import urllib3

urllib3.disable_warnings()
# 进度条库
from tqdm import tqdm
import os

# https://image.baidu.com/search/acjson中的headers和cookies
cookies = {
    'BDqhfp': '%E7%8B%97%E7%8B%97%26%26NaN-1undefined%26%2618880%26%2621',
    'BIDUPSID': '06338E0BE23C6ADB52165ACEB972355B',
    'PSTM': '1646905430',
    'BAIDUID': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
    'H_PS_PSSID': '35836_35105_31254_36024_36005_34584_36142_36120_36032_35993_35984_35319_26350_35723_22160_36061',
    'BDSFRCVID': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
    'H_BDCLCKID_SF': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'BDSFRCVID_BFESS': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
    'H_BDCLCKID_SF_BFESS': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'indexPageSugList': '%5B%22%E7%8B%97%E7%8B%97%22%5D',
    'cleanHistoryStatus': '0',
    'BAIDUID_BFESS': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
    'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
    'ab_sr': '1.0.1_Y2YxZDkwMWZkMmY2MzA4MGU0OTNhMzVlNTcwMmM2MWE4YWU4OTc1ZjZmZDM2N2RjYmVkMzFiY2NjNWM4Nzk4NzBlZTliYWU0ZTAyODkzNDA3YzNiMTVjMTllMzQ0MGJlZjAwYzk5MDdjNWM0MzJmMDdhOWNhYTZhMjIwODc5MDMxN2QyMmE1YTFmN2QyY2M1M2VmZDkzMjMyOThiYmNhZA==',
    'delPer': '0',
    'PSINO': '2',
    'BA_HECTOR': '8h24a024042g05alup1h3g0aq0q',
}


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
}




def craw_single_class(keyword, DOWNLOAD_NUM=200):
    # 爬取文件存盘
    if os.path.exists('dataset/' + keyword):
        print('图片将存入dataset/{}'.format(keyword), "中")
    else:
        os.makedirs('dataset/{}'.format(keyword))
        print('新建文件夹：dataset/{}'.format(keyword))
    count = 1
    with tqdm(total=DOWNLOAD_NUM, position=0, leave=True) as pbar:
        # 爬取第几张
        num = 0
        # 是否继续爬取
        FLAG = True
        while FLAG:
            page = 30 * count
            params = (
                ('tn', 'resultjson_com'),
                ('logid', '12508239107856075440'),
                ('ipn', 'rj'),
                ('ct', '201326592'),
                ('is', ''),
                ('fp', 'result'),
                ('fr', ''),
                ('word', f'{keyword}'),
                ('queryWord', f'{keyword}'),
                ('cl', '2'),
                ('lm', '-1'),
                ('ie', 'utf-8'),
                ('oe', 'utf-8'),
                ('adpicid', ''),
                ('st', '-1'),
                ('z', ''),
                ('ic', ''),
                ('hd', ''),
                ('latest', ''),
                ('copyright', ''),
                ('s', ''),
                ('se', ''),
                ('tab', ''),
                ('width', ''),
                ('height', ''),
                ('face', '0'),
                ('istype', '2'),
                ('qc', ''),
                ('nc', '1'),
                ('expermode', ''),
                ('nojc', ''),
                ('isAsync', ''),
                ('pn', f'{page}'),
                ('rn', '30'),
                ('gsm', '1e'),
                ('1647838001666', ''),
            )

            response = requests.get('https://image.baidu.com/search/acjson', headers=headers, params=params,cookies=cookies)
            # response = requests.get('https://image.baidu.com/search/acjson', headers=headers)
            print(response.status_code,keyword)
            if response.status_code == 200:
                try:
                    json_data = response.json().get("data")
                    print(json_data)
                    if json_data:
                        for x in json_data:
                            type = x.get("type")
                            if type not in ["gif"]:
                                # thumbURL存储了缩略图地址，目的是爬取缩略图
                                img = x.get("thumbURL")
                                # fromPageTitleEnc中存储了图片名字
                                fromPageTitleEnc = x.get("fromPageTitleEnc")
                                try:
                                    resp = requests.get(url=img, verify=False)
                                    time.sleep(1)
                                    # print(f"链接 {img}")
                                    # print(resp.content)
                                    # 保存文件路径
                                    file_save_path = f'C:/Users/adm/PycharmProjects/Life Below Water/海洋生物图片采集/dataset/{keyword}/{num}.{type}'
                                    f=open(file_save_path,'wb')
                                    f.write(resp.content)
                                    # f.flush()
                                    f.close()
                                    print('第 {} 张图像 {} 爬取完成'.format(num, fromPageTitleEnc))
                                    num += 1
                                    pbar.update(1)  # 进度条更新
                                    # 爬取数量达到要求
                                    if num > DOWNLOAD_NUM:
                                        FLAG = False
                                        print('{} 张图像爬取完毕'.format(num))
                                        break
                                except Exception:
                                    print('写入失败')

                except:
                    pass
            else:
                break

            count += 1



if __name__ == '__main__':
    # 读取海洋生物名称,添加到种类列表
    # with open('海洋生物名称.txt','r',encoding='utf-8') as fname:
    #     allList=fname.readlines()
    #     class_list=[]
    #     for list in allList:
    #         # 去空格换行符
    #         list=list.strip("\n")
    #         list=list.strip("")
    #         # 去重复
    #         list=set(list.split(', '))
    #         class_list.extend(list)
    #     print(class_list)
    class_list=['鳄鱼']
    for each in class_list:
        craw_single_class(each, DOWNLOAD_NUM=200)
