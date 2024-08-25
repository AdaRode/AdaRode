from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
# from XLnet_Adapter import *
from DetANN_Adapter import *
from DeepXSS_Adapter import *
from GraphXSS_Adapter import *
from Sec2vec_Adapter import *
from Tools.Resultsaver import Results
import pandas as pd
import os
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import yaml

# PIK
# model_name='DetANN-DBN' ok
# model_name="DetANN-BiGRU" ok 
# model_name="DetANN-BiLSTM" ok
# model_name="DetANN-RF" ok
# model_name="DetANN-XGB" ok  

# model_name="DeepXSS-BiGRU" ok
# model_name="DeepXSS-BiLSTM" ok
# model_name="DeepXSS-DBN" ok
# model_name="TR-IDS" ok
# model_name="C-BLA" ok

# model_name="GraphXSS" ok

# model_name='RoBERTa' ok

# HPD
# model_name='DetANN-DBN' ok
# model_name="DetANN-BiGRU" ok
# model_name="DetANN-BiLSTM" ok
# model_name="DetANN-RF" ok
# model_name="DetANN-XGB" ok

# model_name="DeepXSS-BiGRU" ok
# model_name="DeepXSS-BiLSTM" ok
# model_name="DeepXSS-DBN" ok
# model_name="TR-IDS" ok
# model_name="C-BLA" ok

# model_name='RoBERTa' ok
# model_name='XLNet' run
model_name='Ada'


# model_name="GraphXSS"
# dataset='PIK'
dataset='HPD'




def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class RS:
    def __init__(self, config, testset, labelset, typeset, resultsaver,process_id):
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.typeset = typeset
        if model_name.startswith('DeepXSS') or model_name=='TR-IDS' or model_name=='C-BLA':
            self.victim = DeepXSSAdapter(model_name=model_name,dataset=dataset)
        if model_name.startswith('DetANN'):
            self.victim = DNNAdapter(model_name=model_name,dataset=dataset)
        if model_name.startswith('Graph'):
            self.victim = GraphXSSAdapter(model_name=model_name,dataset=dataset)
        if model_name=='RoBERTa' or model_name=='XLNet' or model_name=='Ada':
            self.victim = Sec2vecAdapter(model_name=model_name,dataset=dataset)
        self.maxiter = config['parameters']['max_iterations']
        self.patience = config['parameters']['patience']
        self.resultsaver = resultsaver
        self.xsscount = 1
        self.sqlcount = 1
        self.xss_succ = 0
        self.sql_succ = 0
        self.xss_iter = []
        self.sql_iter = []
        self.processid = process_id
          
    def _filter_attack_order(self,data:str):
        probs = self.victim.get_prob(data)
        # TODO：写完整的筛选算法
        flag = "selected"
        return flag
    
    def _replaceWords(self, _vic:str, _viclabel:str, datatype = "xss",
                      _prob_threshold=0.95, _candi_mode="allin"):

        assert _candi_mode.lower() in ["allin"]

        _candi_tokens = [_vic]
        # candi_labels = [viclabel]
        # 选择候选样本中方法
        if _candi_mode == "allin":
            # 使用随即方法获得一个候选变量
            
            if datatype == "xss":
                Attacker = XssFuzzer(_vic)
                for num in range(21):
                    resflag = Attacker.fuzz(num)
                    if resflag == -1:
                        Attacker.reset()
                        continue
                    _candi_tokens.append(Attacker.current())
                    Attacker.reset()
            else:
                Attacker = SqlFuzzer(_vic)
                for num in range(12):
                    Attacker.fuzz(num)
                    _candi_tokens.append(Attacker.current())
                    Attacker.reset()
                    
            _candi_tokens = _candi_tokens
            _candi_tokens.append(_vic)
            preds = [self.victim.get_pred(sample)[0] for sample in _candi_tokens]

            for i in range(len(_candi_tokens)):
                if preds[i] != _viclabel:
                    records = {"status": "s", "choices": 1, "tokens": _candi_tokens[i],
                            "old_object": _candi_tokens[0], "new_object": _candi_tokens[i],
                            "old_pred": preds[0], "new_pred": preds[i],
                            "adv_x": _candi_tokens[i]}
                    return records
                
            random_decision = random.choice([True, False])

    
            candi_idx = random.randint(0,len(_candi_tokens))
            choices = candi_idx-1
            if random_decision:
                records = {"status": "r", "choices": choices, "tokens": _candi_tokens[choices],
                        "old_object": _candi_tokens[0], "new_object": _candi_tokens[choices],
                        "old_pred": preds[0], "new_pred": preds[choices]}
                return records
            else:
                # TODO:缺一个jump up的算法
                try:
                    if _candi_tokens[0] == _candi_tokens[choices]:
                        records = {"status": "f", "choices": choices, "tokens": _candi_tokens[choices],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[choices],
                                "old_pred": preds[0], "new_pred": preds[choices]}
                        return records
                    else:
                        records = {"status": "a", "choices": choices, "tokens": _candi_tokens[choices],
                                "old_object": _candi_tokens[0], "new_object": _candi_tokens[choices],
                                "old_pred": preds[0], "new_pred": preds[choices]}
                except:
                    records = {"status": "f", "choices": choices, "tokens": _vic,
                                "old_object": _vic, "new_object": _vic,
                                "old_pred": preds[0], "new_pred": preds[0]}
                return records

    
    def rands(self, vic:str, viclabel:str, datatype = "xss", _max_iter=100):
        
        if len(vic) <= 0:
            return {'succ': False, 'tokens': None, 'raw_tokens': None}
        tokens = vic
        jumpup = 0
        if datatype == "xss":self.xsscount+=1
        if datatype == "sql":self.sqlcount+=1
        for iteration in range(1, 1+_max_iter):
            res = self._replaceWords(_vic=tokens, _viclabel=viclabel,  datatype=datatype)
            self._recordResnote(_iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() == 'f':
                jumpup+=1
            if jumpup>self.patience:
                jumpup = 0
                break
            if res['status'].lower() in ['s', 'a']:
                jumpup = 0
                tokens = res['tokens']
                if res['status'].lower() == 's':
                    if datatype == "xss":
                        self.xss_succ+=1
                        self.xss_iter.append(iteration)
                    if datatype == "sql":
                        self.sql_succ+=1
                        self.sql_iter.append(iteration)
                    return {'succ': True, 'tokens': tokens,
                            'raw_tokens': tokens}
        if datatype == "xss":self.xss_iter.append(iteration)
        if datatype == "sql":self.sql_iter.append(iteration)   
        return {'succ': False, 'tokens': None, 'raw_tokens': None}
    
    def _recordResnote(self, _iter=None, _res=None, _prefix="  => "):
        # print("111111111111",_res)
        if _res['status'].lower() == 's':   # Accepted & successful
        
            results = "%s iter %d, SUCC! %s => %s (%d => %d, a=%.5f)" %(
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                     _res['choices'])
            self.resultsaver.savelogDData(info=results)

        elif _res['status'].lower() == 'r':  # Rejected
            results = "%s iter %d, REJ. %s => %s (%d => %d, a=%.5f)" %(
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                     _res['choices'])
            self.resultsaver.savelogDData(info=results)
        
        elif _res['status'].lower() == 'a':  # Accepted
            results = "%s iter %d, ACC! %s => %s (%d => %d, a=%.5f)" %(
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                     _res['choices'])
            self.resultsaver.savelogDData(info=results)
        elif _res['status'].lower() == 'f':  # f
            results = "%s iter %d, ACC! %s => %s (%d => %d, a=%.5f)" %(
                    _prefix, _iter, _res['old_object'], _res['new_object'],
                    _res['old_pred'], _res['new_pred'],
                     _res['choices'])
            self.resultsaver.savelogDData(info=results)

    def getrecordMetrics(self):
        log = "Immediately Record"+"\n"
        log += "="*80 +"\n"
        log += "Current SQL attacks: {}, SQL ASR = {:.2f}%".format(self.sql_succ, (self.sql_succ / self.sqlcount) * 100)+"\n"
        log += "Current XSS attacks: {}, XSS ASR = {:.2f}%".format(self.xss_succ, (self.xss_succ / self.xsscount) * 100)+"\n"
        log += "Current SQL num: {}".format(self.sqlcount)+"\n"
        log += "Current XSS num: {}".format(self.xsscount)+"\n"
        if len(self.sql_iter)>0:averageSQL = sum(self.sql_iter) / len(self.sql_iter)
        else:averageSQL=-1
        if len(self.xss_iter)>0:averageXSS = sum(self.xss_iter) / len(self.xss_iter)
        else:averageXSS=-1
        log += "averageSQL: {}".format(averageSQL)+"\n"
        log += "averageXSS: {}".format(averageXSS)+"\n"
        log += "="*80 +"\n"
        self.resultsaver.savelogDData(info=log)


    def jud_type(self,vicstr:str,iteration:int)->str:
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
        
    def exec(self):
        res = {"ori_raw": [], "ori_label": [], "adv_raw": [], "adv_label": []}
        for iteration, strload in tqdm(enumerate(self.testset), total=len(self.testset),position=self.processid + 1,desc=f"Process {self.processid}"):
            if (iteration) % 10 == 5:
                self.getrecordMetrics()
                print(f"current sql ASR = {self.sql_succ/self.sqlcount:.2f}")
                print(f"current xss ASR = {self.xss_succ/self.xsscount:.2f}")
                with open("./Attackers/AdvMut_data/mid_"+model_name+'.pickle', "wb") as file:
                    pickle.dump(res, file)
            # print("\nEXAMPLE " + str(iteration) + "...")
            self.resultsaver.savelogDData(info="\nEXAMPLE " + str(iteration) + "...")
            start_time = time.time()
            data_a_type = self.jud_type(strload, iteration)
            # self.maxiter = 5
            _res = self.rands(vic=strload, viclabel=self.labelset[iteration],
                            datatype=data_a_type, _max_iter=self.maxiter)
            if _res['succ']:
                # Bypass successfully
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " SUCCEEDED!")
                timecost = "time cost = %.2f min" % ((time.time() - start_time) / 60)
                self.resultsaver.savelogDData(info=timecost)
                res['adv_label'].append(self.labelset[iteration])
                res['adv_raw'].append(_res['raw_tokens'])
                res["ori_raw"].append(strload)
                res["ori_label"].append(self.labelset[iteration])
            else:
                # print("EXAMPLE " + str(iteration) + " FAILED...")
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
    
    resultsaver = Results(f"te{len(data_slice)}_{process_id}_AdaA", "RandomSample", "SIK")
    attacker = RS(config, data_slice, label_slice, type_slice, resultsaver, process_id)
    advdata = attacker.exec()
    return advdata
    

    


    
if __name__ == "__main__":
    # 加载配置文件
    config = load_config("Config/adv_test.yaml")

    # 读取数据
    test_data = pd.read_csv(config['paths'][dataset]['data'])

    # 获取所有数据
    data_list = test_data['Text'].tolist()
    label_list = test_data["Label"].tolist()
    type_list = test_data["type"].tolist()
    # data_list = test_data['Text'].tolist()[14:20]
    # label_list = test_data["Label"].tolist()[14:20]
    # type_list = test_data["type"].tolist()[14:20]

    # 直接运行处理函数，传递所有数据
    advdata = run_process(0, data_list, label_list, type_list, config)

    # 保存合并后的数据
    with open(config['paths'][dataset]['augmented_data_save_path']+model_name+'.pickle', "wb") as file:
        pickle.dump(advdata, file)
    
    print(config['paths'][dataset]['augmented_data_save_path']+model_name+'.pickle')
