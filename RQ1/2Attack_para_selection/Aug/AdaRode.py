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

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class MHM:
    def __init__(self, config, testset, labelset, typeset, resultsaver, process_id):
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.typeset = typeset
        self.victim = RodeXLAdapter()
        self.maxiter = config['parameters']['max_iterations']
        self.patience = config['parameters']['patience']
        self.accept = config['parameters']['accept_rate']
        self.resultsaver = resultsaver
        self.xsscount = 1
        self.sqlcount = 1
        self.xss_succ = 0
        self.sql_succ = 0
        self.xss_iter = []
        self.sql_iter = []
        self.processid = process_id
          
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
    
    def mcmc(self, vic:str, viclabel:str, datatype="xss", _prob_threshold=0.95, _max_iter=100):
        if len(vic) <= 0:
            return {'succ': False, 'tokens': None, 'raw_tokens': None}
        tokens = vic
        jumpup = 0
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
                    if datatype == "xss":
                        self.xss_succ += 1
                        self.xss_iter.append(iteration)
                    if datatype == "sql":
                        self.sql_succ += 1
                        self.sql_iter.append(iteration)
                    return {'succ': True, 'tokens': tokens, 'raw_tokens': tokens}
        if datatype == "xss": self.xss_iter.append(iteration)
        if datatype == "sql": self.sql_iter.append(iteration)
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
        log += "=" * 80 + "\n"
        log += "Total attacks: {}\n".format(total_attacks)
        log += "Total successful attacks: {}\n".format(total_success)
        log += "Overall ASR: {:.2f}%\n".format(overall_asr)
        log += "Overall average iterations: {}\n".format(overall_average_iter)
        log += "=" * 80 + "\n"
        
        self.resultsaver.savelogDData(info=log)

    
    def exec(self):
        res = {"ori_raw": [], "ori_label": [], "adv_raw": [], "adv_label": []}
        for iteration, strload in tqdm(enumerate(self.testset), total=len(self.testset), position=self.processid + 1, desc=f"Process {self.processid}"):
            if (iteration) % 10 == 5:
                self.getrecordMetrics()
                print(f"current sql ASR = {self.sql_succ/self.sqlcount:.2f}")
                print(f"current xss ASR = {self.xss_succ/self.xsscount:.2f}")
                root_path = self.config['paths']['mid_save_path'] + self.config['paths']['dataname']
                os.makedirs(root_path, exist_ok=True)
                with open(root_path + os.sep + "adv_mid{}.pickle".format(self.processid), "wb") as file:
                    pickle.dump(res, file)
            self.resultsaver.savelogDData(info="\nEXAMPLE " + str(iteration) + "...")
            start_time = time.time()
            data_a_type = self.jud_type(strload, iteration)

            _res = self.mcmc(vic=strload, viclabel=self.labelset[iteration], datatype=data_a_type, _max_iter=self.maxiter)
            if _res['succ']:
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " SUCCEEDED!")
                timecost = "time cost = %.2f min" % ((time.time() - start_time) / 60)
                self.resultsaver.savelogDData(info=timecost)
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
        self.getrecordMetrics()
        return res

def run_process(process_id, data_slice, label_slice, type_slice, config):
    device = torch.device(config['parameters']['device'])
    torch.cuda.set_device(device)
    
    resultsaver = Results(f"te{len(data_slice)}_{process_id}", "mhmattacker", "PIK")
    attacker = MHM(config, data_slice, label_slice, type_slice, resultsaver, process_id)
    advdata = attacker.exec()
    
    with open(config['paths']['augmented_data_save_path'].replace('.pickle', f'adv_{process_id}.pickle'), "wb") as file:
        pickle.dump(advdata, file)
    with open(config['paths']['record_save_path'].replace('.pickle', f'_{process_id}.pickle'), "wb") as file:
        pickle.dump(attacker, file)

if __name__ == "__main__":
    config = load_config("Config/adv_config.yaml")
    test_data = pd.read_csv(config['paths']['data'])
    
    data_list = test_data['Text'].tolist()
    label_list = test_data["Label"].tolist()
    type_list = test_data["type"].tolist()

    resultsaver = Results("te_single_thread", "mhmattacker", "PIK")
    attacker = MHM(config, data_list, label_list, type_list, resultsaver, 0)
    advdata = attacker.exec()
    
    with open(config['paths']['augmented_data_save_path'], "wb") as file:
        pickle.dump(advdata, file)
    with open(config['paths']['record_save_path'], "wb") as file:
        pickle.dump(attacker, file)
