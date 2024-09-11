from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
from XLnet_Adapter import *
from Tools.Resultsaver import Results
import pandas as pd
import os
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from collections import Counter
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class MHM:
    def __init__(self, config, testset, labelset, resultsaver, process_id):
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.victim = RodeXLAdapter(modelpath=config['paths']['model_checkpoint'])
        self.maxiter = config['parameters']['max_iterations']
        self.patience = config['parameters']['patience']
        self.accept = config['parameters']['accept_rate']
        self.resultsaver = resultsaver
        self.xsscount = 0
        self.sqlcount = 0
        self.xss_succ = 0
        self.sql_succ = 0
        self.xss_iter = []
        self.sql_iter = []
        self.succ_iter = []
        self.evasion_count = 0
        self.evasion_succ = 0
        self.attack_count = 0
        self.processid = process_id
        self.sql_choice = []
        self.xss_choice = []   
        self.sql_selection = [] 
        self.xss_selection = [] 
          
    def _filter_attack_order(self, data:str):
        probs = self.victim.get_prob(data)
        # TODO：写完整的筛选算法
        flag = "selected"
        return flag
    
    def jud_type(self, vicstr:str, iteration:int) -> str:
        try:
            return self.typeset[iteration]
        except:
            # 常见的SQL关键字
            sql_keywords = [
                r"(?i)\bselect\b",
                r"(?i)\bfrom\b",
                r"(?i)\bwhere\b",
                r"(?i)\binsert\b",
                r"(?i)\binto\b",
                r"(?i)\bvalues\b",
                r"(?i)\bupdate\b",
                r"(?i)\bset\b",
                r"(?i)\bdelete\b",
                r"(?i)\bcreate\b",
                r"(?i)\balter\b",
                r"(?i)\bdrop\b",
                r"(?i)\bjoin\b",
                r"(?i)\binner\b",
                r"(?i)\bleft\b",
                r"(?i)\bright\b",
                r"(?i)\bouter\b",
                r"(?i)\bgroup\b",
                r"(?i)\bby\b",
                r"(?i)\border\b",
                r"(?i)\bhaving\b",
                r"(?i)\bunion\b",
                r"(?i)\bexec\b"
            ]

            # 常见的XSS特征
            xss_patterns = [
                r"(?i)<script.*?>.*?</script.*?>",
                r"(?i)<.*?javascript:.*?>",
                r"(?i)<.*?on\w+=.*?>",
                r"(?i)alert\(",
                r"(?i)document\.cookie",
                r"(?i)<iframe.*?>.*?</iframe.*?>",
                r"(?i)<img.*?src=.*?>",
                r"(?i)<.*?>",
                r"(?i)&lt;.*?&gt;"
            ]

            for pattern in sql_keywords:
                if re.search(pattern, vicstr):
                    return "sql"

            for pattern in xss_patterns:
                if re.search(pattern, vicstr):
                    return "xss"

            return "sql"
    
    def _replaceWords(self, _vic:str, _viclabel:str, datatype="xss", _prob_threshold=0.95, _candi_mode="allin"):
        _prob_threshold=self.accept
        assert _candi_mode.lower() in ["allin"]

        if self._filter_attack_order(_vic) == "selected":
            candi_tokens = [_vic]
            if _candi_mode == "allin":
                if datatype == "xss":
                    Attacker = XssFuzzer(_vic)
                    for num in range(21):
                        resflag = Attacker.fuzz(num)
                        if resflag == -1:
                            Attacker.reset()
                            continue
                        candi_tokens.append(Attacker.current())
                        Attacker.reset()
                elif datatype == "sql":
                    Attacker = SqlFuzzer(_vic)
                    for num in range(12):
                        Attacker.fuzz(num)
                        candi_tokens.append(Attacker.current())
                        Attacker.reset()
                else:
                    print("GG!")
                    return 0
                _candi_tokens = candi_tokens
                candi_tokens.append(_vic)
                probs = np.array([self.victim.get_prob(sample) for sample in _candi_tokens])
                preds = [self.victim.get_pred(sample)[0] for sample in _candi_tokens]

                for i in range(len(candi_tokens)):
                    if preds[i] != _viclabel:
                        records = {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[i],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[i][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[i],
                                "adv_x": candi_tokens[i]}
                        return records
                
                candi_idx = np.argmin(probs[1:,_viclabel]) + 1
                candi_idx = int(candi_idx)
                if datatype == "xss":
                    self.xss_choice.append(candi_idx)
                elif datatype == "sql":
                    self.sql_choice.append(candi_idx)
                alpha = (1 - probs[candi_idx][_viclabel] + 1e-10) / (1 - probs[0][_viclabel] + 1e-10)

                if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                    records = {"status": "r", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                            "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                            "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                            "old_pred": preds[0], "new_pred": preds[candi_idx]}
                    return records
                else:
                    if _candi_tokens[0] == _candi_tokens[candi_idx]:
                        records = {"status": "f", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[candi_idx]}
                        return records
                    else:
                        records = {"status": "a", "alpha": alpha, "tokens": candi_tokens[candi_idx],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[candi_idx],
                                "old_prob": probs[0][_viclabel], "new_prob": probs[candi_idx][_viclabel],
                                "old_pred": preds[0], "new_pred": preds[candi_idx]}
                    return records
        else:
            print("Not regular")
    
    def mcmc(self, vic:str, viclabel:int, datatype="xss", _prob_threshold=0.95, _max_iter=100):
        if len(vic) <= 0:
            return {'succ': False, 'tokens': None, 'raw_tokens': None}
        tokens = vic
        jumpup = 0
        if viclabel!=0:
            self.evasion_count += 1
        if datatype == "xss": self.xsscount += 1
        if datatype == "sql": self.sqlcount += 1
        for iteration in range(1, 1 + _max_iter):
            res = self._replaceWords(_vic=tokens, _viclabel=viclabel, datatype=datatype, _prob_threshold=_prob_threshold)
            self._recordResnote(_iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() == 'f':
                jumpup += 1
            if jumpup > self.patience:
                jumpup = 0
                break
            if res['status'].lower() in ['s', 'a']:
                jumpup = 0
                tokens = res['tokens']
                if res['status'].lower() == 's':
                    if viclabel!=0:
                        self.evasion_succ += 1
                    if datatype == "xss":
                        self.xss_succ += 1
                    if datatype == "sql":
                        self.sql_succ += 1
                    self.succ_iter.append(iteration)
                    self.sql_selection += self.sql_choice
                    self.xss_selection += self.xss_choice
                    self.sql_choice = []
                    self.xss_choice = []
                    return {'succ': True, 'tokens': tokens, 'raw_tokens': tokens}
        if datatype == "xss": self.xss_iter.append(iteration)
        if datatype == "sql": self.sql_iter.append(iteration)
        self.sql_choice = []
        self.xss_choice = []
        return {'succ': False, 'tokens': None, 'raw_tokens': None}
    
    def _recordResnote(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':
            results = "%s iter %d, SUCC! %s => %s (%d => %d, %.9f => %.9f) a=%.5f" % (
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                    _res['old_prob'], _res['new_prob'], _res['alpha'])
            self.resultsaver.savelogDData(info=results)
        elif _res['status'].lower() == 'r':
            results = "%s iter %d, REJ. %s => %s (%d => %d, %.9f => %.9f) a=%.5f" % (
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                    _res['old_prob'], _res['new_prob'], _res['alpha'])
            self.resultsaver.savelogDData(info=results)
        elif _res['status'].lower() == 'a':
            results = "%s iter %d, ACC! %s => %s (%d => %d, %.9f => %.9f) a=%.5f" % (
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                    _res['old_prob'], _res['new_prob'], _res['alpha'])
            self.resultsaver.savelogDData(info=results)
        elif _res['status'].lower() == 'f':
            results = "%s iter %d, ACC! %s => %s (%d => %d, %.9f => %.9f) a=%.5f" % (
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                    _res['old_prob'], _res['new_prob'], _res['alpha'])
            self.resultsaver.savelogDData(info=results)

    def getrecordMetrics(self):
        # Calculate individual metrics
        sql_asr = (self.sql_succ / self.sqlcount) * 100 if self.sqlcount > 0 else 0
        xss_asr = (self.xss_succ / self.xsscount) * 100 if self.xsscount > 0 else 0
        averageSQL = sum(self.sql_iter) / len(self.sql_iter) if len(self.sql_iter) > 0 else -1
        averageXSS = sum(self.xss_iter) / len(self.xss_iter) if len(self.xss_iter) > 0 else -1
        
        # Calculate overall metrics
        total_attacks = self.sqlcount + self.xsscount
        total_success = self.sql_succ + self.xss_succ
        overall_asr = (total_success / total_attacks) * 100 if total_attacks > 0 else 0
        overall_average_iter = (sum(self.sql_iter) + sum(self.xss_iter)) / (len(self.sql_iter) + len(self.xss_iter)) if (len(self.sql_iter) + len(self.xss_iter)) > 0 else -1
        
        log = "Immediately Record\n"
        log += "=" * 80 + "\n"
        log += "Current SQL attacks: {}, SQL ASR = {:.2f}%\n".format(self.sql_succ, sql_asr)
        log += "Current XSS attacks: {}, XSS ASR = {:.2f}%\n".format(self.xss_succ, xss_asr)
        log += "Current SQL num: {}\n".format(self.sqlcount)
        log += "Current XSS num: {}\n".format(self.xsscount)
        log += "Average SQL iterations: {}\n".format(averageSQL)
        log += "Average XSS iterations: {}\n".format(averageXSS)
        log += "Total attacks: {}\n".format(total_attacks)
        log += "Total successful attacks: {}\n".format(total_success)
        log += "Overall ASR: {:.2f}%\n".format(overall_asr)
        log += "Overall average iterations: {}\n".format(overall_average_iter)
        log += "=" * 80 + "\n"
        
        self.resultsaver.savelogDData(info=log)

    def count_elements(self, input_list):
        # 使用 Counter 统计元素出现的次数
        element_count = Counter(input_list)
        # 将 Counter 对象转换为字典
        return dict(element_count)
        
    def result_caculate(self):
        result = {
            "overall": {  # 添加一个总体结果字典
                "ASR": 0,
                "SAQ": 0,  # 平均成功攻击查询数
                "TAQ": 0,  # 平均查询步数
                "EVN": 0,  # 恶意逃逸计数
                "ER": 0,   # 恶意逃逸比率
            },
            "sql": {
                "AF": {}  # 填充SQL攻击类型的详细统计信息
            },
            "xss": {
                "AF": {}  # 填充XSS攻击类型的详细统计信息
            }
        }

        # Calculate overall metrics
        total_attacks = self.sqlcount + self.xsscount
        total_success = self.sql_succ + self.xss_succ
        overall_asr = (total_success / total_attacks) * 100 if total_attacks > 0 else 0
        result["overall"]["ASR"] = overall_asr  # 填充总体ASR

        # 计算成功攻击的平均查询数（SAQ）
        success_average_queries = np.mean(self.succ_iter) if self.succ_iter else 0
        result["overall"]["SAQ"] = success_average_queries  # 填充总体SAQ

        # 计算所有攻击的平均查询步数（TAQ）
        total_average_queries = (sum(self.sql_iter) + sum(self.xss_iter)) / (len(self.sql_iter) + len(self.xss_iter)) if (len(self.sql_iter) + len(self.xss_iter)) > 0 else -1
        result["overall"]["TAQ"] = total_average_queries  # 填充总体TAQ

        # 恶意逃逸统计
        evasion_value = self.evasion_succ if self.evasion_succ > 0 else 0
        result["overall"]["EVN"] = evasion_value  # 填充总体EVN

        # 计算恶意逃逸比率
        evasion_ratio = (self.evasion_succ / total_attacks) * 100 if total_attacks > 0 else 0
        result["overall"]["ER"] = evasion_ratio  # 填充总体ER
        print(self.sql_choice)
        print(self.xss_choice)
        # 记录每种攻击类型的详细信息（AF）

        result["sql"]["AF"] = self.count_elements(self.sql_selection)  # 填充SQL攻击类型的详细统计信息
        result["xss"]["AF"] = self.count_elements(self.xss_selection)  # 填充XSS攻击类型的详细统计信息

        # Log the results
        log = "Results Summary\n"
        log += "=" * 80 + "\n"
        log += "Overall Attack Success Rate (ASR): {:.2f}%\n".format(overall_asr)
        log += "Average Attack Queries for Successful Attacks (SAQ): {:.2f}\n".format(success_average_queries)
        log += "Average Steps Required (TAQ): {:.2f}\n".format(total_average_queries)
        log += "Evasion Count (EVN): {}\n".format(evasion_value)
        log += "Evasion Ratio (ER): {:.2f}%\n".format(evasion_ratio)
        log += "Total Attack Count: {}\n".format(self.attack_count)
        log += "=" * 80 + "\n"

        self.resultsaver.savelogDData(info=log)

        return result  # 返回计算结果


    
    def exec(self):
        res = {"ori_raw": [], "ori_label": [], "adv_raw": [], "adv_label": []}
        start_time = time.time()  # 记录开始时间
        for iteration, strload in tqdm(enumerate(self.testset), total=len(self.testset), position=self.processid + 1, desc=f"Process {self.processid}"):
            # 检查是否超过一小时
            if time.time() - start_time >= 3600:  # 3600秒 = 1小时
                print("时间超过一小时，强制中断计算...")
                metrics = self.result_caculate()  # 计算指标
                print(metrics)
                with open(self.config['paths']['attack_result_save_path'] + dataset + "_ada{}.json".format(self.processid), "w", encoding='utf-8') as file:
                    json.dump(metrics, file, ensure_ascii=False, indent=4)

                return res  # 返回已经计算的结果

            if (iteration) % 10 == 5:
                self.getrecordMetrics()
                print(f"current sql ASR = {self.sql_succ/self.sqlcount:.2f}" if self.sqlcount != 0 else "current sql ASR = N/A")
                print(f"current xss ASR = {self.xss_succ/self.xsscount:.2f}" if self.xsscount != 0 else "current xss ASR = N/A")

            self.resultsaver.savelogDData(info="\nEXAMPLE " + str(iteration) + "...")
            data_a_type = self.jud_type(strload, iteration)

            _res = self.mcmc(vic=strload, viclabel=self.labelset[iteration], datatype=data_a_type, _max_iter=self.maxiter)
            if _res['succ']:
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " SUCCEEDED!")
                res['adv_label'].append(self.labelset[iteration])
                res['adv_raw'].append(_res['raw_tokens'])
                res["ori_raw"].append(strload)
                res["ori_label"].append(self.labelset[iteration])
            else:
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " FAILED...")
                res['adv_raw'].append(strload)
                res['adv_label'].append(self.labelset[iteration])
                res["ori_raw"].append(strload)
                res["ori_label"].append(self.labelset[iteration])
            self.attack_count += 1
        
        self.getrecordMetrics()
        return res



if __name__ == "__main__":
    config = load_config("Config/adv_config_PIK.yaml")
    dataset = config['parameters']["dataname"]
    

    for process in range(1,11):
        test_data = pd.read_csv(os.getcwd()+os.sep+"Data"+os.sep+dataset+os.sep+f"shuffled_data_part_{process}.csv")
        print(os.getcwd()+os.sep+"Data"+os.sep+dataset+os.sep+f"shuffled_data_part_{process}.csv"+" has been loaded...")
        data_list = test_data['Text'].tolist()
        label_list = test_data["Label"].tolist()

        resultsaver = Results("te_single_thread", "mhmattacker", "PIK")


        attacker = MHM(config, data_list, label_list, resultsaver, process)
        advdata = attacker.exec()
        print("Process{} have finished".format(process))
    


