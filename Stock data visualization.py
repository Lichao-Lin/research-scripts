import json
import os
import pandas as pd
import requests
# import openpyxl

session = requests.session()

# 数据格式化
def pd_unit(newArr):
    # if float(newArr) >= 100000000 or float(newArr) <= -100000000:
    #     result = format(float(newArr) / 100000000, '.2f') + '亿'
    # else:
    result = format(float(newArr) / 10000, '.2f')
    return result

# 获取K线数据
def getLinesData(code: str, id: str):
    # beg和end可以调整数据区间
    params = f"fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&" \
             "fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&" \
             "beg=20180101&" \
             "end=20500101&" \
             "ut=fa5fd1943c7b386f172d6893dbfba10b&" \
             "rtntype = 6&" \
             f"secid={code}.{id}&" \
             "klt=101&" \
             "fqt=1&"

    res = session.get("http://push2his.eastmoney.com/api/qt/stock/kline/get", params=params)
    data = json.loads(res.text)
    return data["data"]["klines"]

# 获取公司数据
def getCompanyData(code: str, id: str):
    area = ''
    if code == "0":
        area = "SZ"
    elif code == "1":
        area = "SH"
    res = session.get(f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={area}{id}")
    result = json.loads(res.text)
    cData = []
    cData.append(result["jbzl"][0]["PROVINCE"])
    cData.append(result["jbzl"][0]["SECURITY_NAME_ABBR"])
    return cData

# 获取股票概念
def getConcept(code: str, id: str):
    params = f"fltt=1&"\
             "invt=2&"\
             "fields=f14%2Cf12%2Cf13%2Cf3%2Cf152%2Cf4%2Cf128%2Cf140%2Cf141&"\
             f"secid={code}.{id}&"\
             "ut=fa5fd1943c7b386f172d6893dbfba10b&"\
             "pi=0&"\
             "po=1&"\
             "np=1&"\
             "pz=5&"\
             "spt=3"

    res = session.get("http://push2.eastmoney.com/api/qt/slist/get", params=params)
    data = json.loads(res.text)
    result = data["data"]["diff"]
    concept = []
    for item in result:
        concept.append(item["f14"])
    return concept

# 获取股票列表
def getStockList(title: str):
    # pz可以调整爬取股票的数量
    params = "pn=1&"\
             "pz=80&"\
             "po=1&"\
             "np=1&"\
             "ut=bd1d9ddb04089700cf9c27f6f7426281&"\
             "fltt=2&"\
             "invt=2&"\
             "wbp2u=|0|0|0|web&"\
             "fid=f3&"\
             f"fs=b:{title}+f:!50&"\
             "fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45&"\

    res = session.get("http://83.push2.eastmoney.com/api/qt/clist/get", params=params)
    result = json.loads(res.text)
    lists = result["data"]["diff"]
    data = []
    for item in lists:
        stockObj = {"code": item["f13"], "id": item["f12"]}
        data.append(stockObj)
    return data

#打包为xlsx文件
def getxlsx(code: str, id: str):
    cur_dir = os.path.dirname(__file__)
    res = pd.DataFrame(columns=(
    "股票代码", "股票简称", "日期", "开盘", "收盘", "最高", "最低", "涨跌幅", "涨跌额", "成交量(万)", "成交额(万)", "振幅",
    "换手率", "行业", "地区", "概念1", "概念2", "概念3", "概念4"))

    linesData = getLinesData(code, id)
    concept = getConcept(code, id)
    companyData = getCompanyData(code, id)
    # 计数
    count = 1
    for item in linesData:
        newArr = item.split(',')
        count = count + 1
        # 行业 地区 概念
        res.loc[count] = [id, companyData[1], newArr[0], newArr[1], newArr[2], newArr[3], newArr[4], newArr[8] + '%',
                          newArr[9], pd_unit(newArr[5]), pd_unit(newArr[6]), newArr[7] + '%', newArr[10] + '%',
                          concept[0], companyData[0], concept[1], concept[2], concept[3], concept[4]]
        # 日期: newArr[0]
        # 开盘newArr[1] 收盘newArr[2] 最高newArr[3] 最低newArr[4] 涨跌幅newArr[8]+'%' 涨跌额newArr[9] 成交量pd_unit(newArr[5]) 成交额pd_unit(newArr[6]) 振幅newArr[7]+'%'  换手率newArr[10]+'%'

    res.to_excel(os.path.join(cur_dir, id + ".xlsx"))
