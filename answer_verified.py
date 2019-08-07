import re

def get_insurant_person(doc, question):
    """
    ["投保的人是谁", "投保人是谁"]这两类问题的抽取规则
    如果该函数返回为None说明doc中没有符合规则的答案，此时以模型预测为准
    :param doc:
    :return:
    """
    # 骐创公司向太平洋保险公司投保
    q_pattern = re.compile(r'向([^公]{0,20}公司)投保的人是谁')
    q_r = re.findall(q_pattern, question)
    if len(q_r) > 0:
        pattern0 = re.compile(r'([^,，。、；;：:由告按照约定]{2,10})向[^公]{0,20}公司投保')
        r0 = re.findall(pattern0, doc)
        if len(r0) > 0:
            return r0[0]
        else:
            return None
    pattern1 = re.compile(r'投保人为(?:被告)?([^,，。、；;：:的]{2,10})')
    r1 = re.findall(pattern1, doc)
    pattern2 = re.compile(r'([^,，。、；;：:由告按照约定]{2,10})(?:按照约定)?作为投保人')
    r2 = re.findall(pattern2, doc)
    pattern3 = re.compile(r'([^,，。、；;：:驾驶将就为曾以及认定负责任作人所的日与]{2,15})(?:作为被保险人)?[^,，。、；：:人在责日与]*在原告处投保')
    r3 = re.findall(pattern3, doc)
    # (被告韩4)为其所有的xx向原告xx公司投保了xx险
    pattern4 = re.compile(r'([^,，。、；;：:人]{2,15})[为就]其[^向]+向原告[^投]{2,20}投保')
    r4 = re.findall(pattern4, doc)
    r_total = r1 + r2 + r3 + r4
    # print("doc:", doc)
    if len(r_total) > 0:
        # print(r_total[0])
        return r_total[0]
    else:
        return None


def get_insurant_company(doc):
    """
    "向什么公司投保"类问题的抽取规则，直接抽取xxx公司作为最终答案
    如果该函数返回为None说明doc中没有符合规则的答案，此时以模型预测为准
    :param doc:
    :param question:
    :return:
    """
    pattern1 = re.compile(r'约定([^公]{0,20}公司[^公共]{0,20}(?:公司)?)(?:共同)?为[^承]{1,20}承保')
    r1 = re.findall(pattern1, doc)
    pattern2 = re.compile(r'由([^公]{0,20}公司)承保')
    r2 = re.findall(pattern2, doc)
    pattern3 = re.compile(r'[原被]告([^公]{0,15}公司(?:[^支]{1,20}支公司)?)作为[^保]{1,20}保险人')
    r3 = re.findall(pattern3, doc)
    pattern4 = re.compile(r'[向在](?:保险人)?(?:[原被]告)?([^公]{0,20}公司(?:[^支]{1,20}支公司)?)处?(?:申请)?(?:进行)?了?投保')
    r4 = re.findall(pattern4, doc)
    pattern5 = re.compile(r'[向在](?:[原被]告)?([^公]{0,20}公司(?:[^支]{1,20}支公司)?)处?购买了?[^险]{1,20}险')
    r5 = re.findall(pattern5, doc)
    pattern6 = re.compile(r'被告([^公]{0,15}公司(?:[^支]{1,20}支公司)?)赔偿')
    r6 = re.findall(pattern6, doc)
    r_total = r1 + r2 + r3 + r4 + r5 + r6
    # print("doc:", doc)
    if len(r_total) > 0:
        return r_total[0]
    else:
        return None


# def find_long_answer(pred, doc, question):
#     """
#     "案件发生经过是怎样的"类问题的验证
#     符合三种模式的答案返回True，否则返回False
#     :param pred: 预测的答案
#     :param doc: 对应的文档
#     :param question: 对应的问题
#     :param casename: 对应的casename
#     :return: True or False
#     """
#     if question.find('案件发生经过是怎样的') >= 0:
#         q_pattern = re.compile(r'([^罪案]+)罪?案件发生经过')
#         q_r = re.findall(q_pattern, question)
#         pattern1 = re.compile(r'经审理查明[，：]?([^。]+。)')
#         pattern2 = re.compile(r'指控[，：]([^。]+。)')
#         r1 = re.findall(pattern1, doc)
#         r2 = re.findall(pattern2, doc)
#         r3 = []
#         max_common_length = 0
#         best_ri_index = -1
#         if len(q_r) > 0:
#             pattern3 = re.compile(r'{}罪|(?:事实)([^。]+。)')
#             r3 = re.findall(pattern3, doc)
#         pred_start_index = doc.find(pred)
#         pred_end_index = pred_start_index + len(pred) - 1
#         r_total = r1 + r2 + r3
#         for i, ri in enumerate(r_total):
#             start_index = doc.find(ri)
#             end_index = start_index + len(ri) - 1
#             if max(start_index, pred_start_index) <= min(pred_end_index, end_index):
#                 index_set = [start_index, pred_start_index, end_index, pred_end_index]
#                 index_set.sort()
#                 common_length = index_set[2] - index_set[1] + 1
#                 if common_length > max_common_length:
#                     max_common_length = common_length
#                     best_ri_index = i
#         if best_ri_index >= 0:
#             if len(r_total[best_ri_index]) > len(pred):
#                 return r_total[best_ri_index]
#             else:
#                 return pred
#         else:
#             return pred


def find_long_answer(pred, doc, question):
    """
    "案件发生经过是怎样的"类问题的验证
    符合三种模式的答案返回True，否则返回False
    :param pred: 预测的答案
    :param doc: 对应的文档
    :param question: 对应的问题
    :param casename: 对应的casename
    :return: True or False
    """
    if question.find('案件发生经过是怎样的') >= 0:
        q_pattern = re.compile(r'([^罪案]+)罪?案件发生经过')
        q_r = re.findall(q_pattern, question)
        pattern1 = re.compile(r'经审理查明[，：]?([^。]+。)')
        pattern2 = re.compile(r'指控[，：]([^。]+。)')
        r1 = re.findall(pattern1, doc)
        r2 = re.findall(pattern2, doc)
        max_common_length = 0
        best_ri_index = -1
        if len(q_r) > 0:
            pattern01 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:综上)')
            pattern02 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:[一二三四五六七八九]、[^罪]{2,10}(?:罪|(?:事实)))')
            pattern03 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:经[^鉴]*鉴定)')

            pattern04 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:具体犯罪事实如下)')
            pattern05 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:至[^日]{1,5}日[0-9]+时)')
            pattern06 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)(?:(?:（[一二三四五六七八九]）)[^事实罪]{2,10}(?:罪|(?:事实)))')

            pattern07 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+?)[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日[，、]')
            pattern08 = re.compile(r'' + q_r[0] + '(?:罪|(?:事实))(.+)')

            r01 = re.findall(pattern01, doc)
            r02 = re.findall(pattern02, doc)
            r03 = re.findall(pattern03, doc)
            r04 = re.findall(pattern04, doc)
            r05 = re.findall(pattern05, doc)
            r06 = re.findall(pattern06, doc)
            r07 = re.findall(pattern07, doc)
            r08 = re.findall(pattern08, doc)
            r0_total = r01 + r02 + r03 + r04 + r05 + r06 + r07 + r08
            if len(r0_total) > 0:
                return r0_total[0]
            else:
                return None
        pred_start_index = doc.find(pred)
        pred_end_index = pred_start_index + len(pred) - 1
        r_total = r1 + r2
        for i, ri in enumerate(r_total):
            start_index = doc.find(ri)
            end_index = start_index + len(ri) - 1
            if max(start_index, pred_start_index) <= min(pred_end_index, end_index):
                index_set = [start_index, pred_start_index, end_index, pred_end_index]
                index_set.sort()
                common_length = index_set[2] - index_set[1] + 1
                if common_length > max_common_length:
                    max_common_length = common_length
                    best_ri_index = i
        if best_ri_index >= 0:
            if len(r_total[best_ri_index]) > len(pred):
                return r_total[best_ri_index]
            else:
                return pred
        else:
            return pred


def find_every_month(doc, question):
    if question.find('甲方每月应支付乙方服务费多少元')>0:
        pass


def find_time_span(question, pred):
    """
    验证“有效时间是多久”类型问题的答案的有效性，对预测的答案返回一个修剪的答案
    :param question:对应的问题
    :param pred:预测的答案
    :return:修剪过后的答案
    """
    if question.find('有效时间是多久') >= 0:
        all_number = "零一二三四五六七八九十"
        pattern1 = re.compile(r'自?[0-9]+年[0-9]+月[0-9]+日[0-9{}]*[时]?[起]?[至到][0-9]+年[0-9]+月[0-9]+日[0-9{}]*时?止?'.
                    format(all_number, all_number))
        pattern2 = re.compile(r'[0-9{}]+年[零又]?[0-9]*个?月?'.format(all_number))
        pattern3 = re.compile(r'[0-9{}]*个月'.format(all_number))
        pattern4 = re.compile(r'自[^起]+起?[，,]?至[^止]+止'.format(all_number))
        p1 = re.findall(pattern1, pred)
        p2 = re.findall(pattern2, pred)
        p3 = re.findall(pattern3, pred)
        p4 = re.findall(pattern4, pred)
        if len(p1) > 0:
            final_text = re.split('\d+$', p1[0])[0]
            return final_text
        elif len(p2) > 0:
            return p2[0]
        elif len(p3) > 0:
            return p3[0]
        elif len(p4) > 0:
            return p4[0]
        # return pred


def repair_time(question, pred):
    if question.find('事故发生时间是什么时候？') >= 0:
        if pred:
            if pred[-1] == '许':
                pred = pred[:-1]
        return pred


def repair_result(doc, question, pred):
    if question.find('事故结果如何') >= 0:
        for v in ['造成', '导致', '致使', '致']:
            if doc.find(v+pred) >= 0:
                return v+pred
    return pred


def with_span(question):
    # 这类问题一定有答案
    if question.find("事故结果如何") >= 0 or question.find("什么机构出具的道路交通事故认定书") >= 0:
        return True
    else:
        return False


# if __name__ == '__main__':
#     strs = ""
#     origin_file = "../data/big_train_data.json"
#     example = answer_rule.extract_question(origin_file)
#     for e in example:
#         if e["answer"] != find_long_answer(e["answer"], e["doc"], e["question"]):
#             print(e["answer"], find_long_answer(e["answer"], e["doc"], e["question"]))
