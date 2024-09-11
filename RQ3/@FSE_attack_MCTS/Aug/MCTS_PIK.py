import time
import random
import math
import pickle
import pandas as pd
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import torch
from XLnet_Adapter import *
from Tools.Resultsaver import Results
from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
import re
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class TargetModel(object):
    def __init__(self, victim, init_payload, resultsaver) -> None:
        self.var_payload_record = init_payload
        self.var_iter = 0
        self.resultsaver = resultsaver
        self.victim = victim
        self.xss_record = [] 
        self.sql_record = []
    
    def jud_type(self, vicstr:str) -> str:
        sql_keywords = [
            r"(?i)\bselect\b", r"(?i)\bfrom\b", r"(?i)\bwhere\b", r"(?i)\binsert\b",
            r"(?i)\binto\b", r"(?i)\bvalues\b", r"(?i)\bupdate\b", r"(?i)\bset\b",
            r"(?i)\bdelete\b", r"(?i)\bcreate\b", r"(?i)\balter\b", r"(?i)\bdrop\b",
            r"(?i)\bjoin\b", r"(?i)\binner\b", r"(?i)\bleft\b", r"(?i)\bright\b",
            r"(?i)\bouter\b", r"(?i)\bgroup\b", r"(?i)\bby\b", r"(?i)\border\b",
            r"(?i)\bhaving\b", r"(?i)\bunion\b", r"(?i)\bexec\b"
        ]

        xss_patterns = [
            r"(?i)<script.*?>.*?</script.*?>", r"(?i)<.*?javascript:.*?>", 
            r"(?i)<.*?on\w+=.*?>", r"(?i)alert\(", r"(?i)document\.cookie", 
            r"(?i)<iframe.*?>.*?</iframe.*?>", r"(?i)<img.*?src=.*?>", 
            r"(?i)<.*?>", r"(?i)&lt;.*?&gt;"
        ]

        for pattern in sql_keywords:
            if re.search(pattern, vicstr):
                return "sql"

        for pattern in xss_patterns:
            if re.search(pattern, vicstr):
                return "xss"

        return "sql"
        
    def evaluate(self, mutated_payload, label, action):
        result = self.victim.get_pred(mutated_payload)[0]
        self.var_iter += 1
        # 判断注入结果是否为正常结果
        if self.jud_type(mutated_payload) == "xss":
            self.xss_record.append(action)
        else:
            self.sql_record.append(action)

        if result != label:
            self._printRes(mutated_payload,"s",self.var_iter,action)
            self.var_iter = 0
            return 1.0
        else:
            self._printRes(mutated_payload,"r",self.var_iter,action)
            return -1.0
    
    def _printRes(self, mut_payload, satus = "r", _iter=None, action = -1, _prefix="  => "):
        if satus == 's':   # Accepted & successful
            results = "%s iter %d, SUCC! %s => %s action choice %s" %(
                    _prefix, _iter, self.var_payload_record, mut_payload, str(action))
            # print(results, flush=True)
            self.var_payload_record = mut_payload
            self.resultsaver.savelogDData(info=results)
        else:
            results = "%s iter %d, FAIL! %s => %s action choice %s" %(
                    _prefix, _iter, self.var_payload_record, mut_payload, str(action))
            # print(results, flush=True)
            self.var_payload_record = mut_payload
            self.resultsaver.savelogDData(info=results)
    def get_action_record(self):
        return self.xss_record,self.sql_record
    def reset_action_record(self):
        self.xss_record = []
        self.sql_record = []

# 定义Node类
class Node(object):
    def __init__(self, action, parent=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.total_reward = 0
        self.num_visits = 0

class MonteCarloTreeSearchFuzzer(object):
    def __init__(self, mutation_num, payload_type, attack_target, label, victim_model,node):
        self.mutation_num = mutation_num
        if payload_type == "xss":
            self.payload_fuzzer = XssFuzzer(attack_target)
        else:
            self.payload_fuzzer = SqlFuzzer(attack_target)
        self.victim = victim_model
        self.label = label
        self.node = node
    
    def get_best_payload(self, action):
        self.payload_fuzzer.fuzz(action)
        best_payload = self.payload_fuzzer.current()
        return best_payload  

    def monte_carlo_tree_search(self, max_step):
        root = self.node
        for _ in range(max_step):
            node = self.selection(root)
            reward = self.simulation(node)
            if reward >= 0:
                return self.get_best_payload(node.action)
            self.backpropagation(node, reward)

        best_child = self.get_best_child(root)
        best_payload = self.get_best_payload(best_child.action)
        # print("YYJ")
        # print(best_child.action)
        # a = input()
        return best_payload

    def selection(self, node):
        while node.children:
            node = self.uct_select_child(node)
        return self.expand(node)

    def expand(self, node):
        for action in range(0, self.mutation_num):
            node.children.append(Node(action, parent=node))
        return random.choice(node.children)

    def uct_select_child(self, node):
        exploration_param = 1 / math.sqrt(2)
        best_child = max(node.children, 
                        key=lambda child: child.total_reward / (child.num_visits + 1e-7) + exploration_param * math.sqrt(2 * math.log(node.num_visits + 1) / (child.num_visits + 1e-7)))
        return best_child

    def simulation(self, node):
        action = node.action
        self.payload_fuzzer.fuzz(action)
        reward = self.evaluate(self.payload_fuzzer.current(),action)
        if self.payload_fuzzer.current().startswith("#API_") and reward < 0:
            # print("="*40+"reset"+"="*40)
            self.payload_fuzzer.reset()
        else:
            self.payload_fuzzer.update()
        return reward

    def backpropagation(self, node, reward):
        while node:
            node.total_reward += reward
            node.num_visits += 1
            node = node.parent

    def get_best_child(self, node):
        best_child = max(node.children, key=lambda child: float('-inf') if child.num_visits == 0 else child.total_reward / child.num_visits)
        return best_child

    def evaluate(self, mutation_payload,action):
        reward = self.victim.evaluate(mutation_payload,self.label,action)
        return reward
    
    def get_fuzzer(self):
        return self.payload_fuzzer

class MCTS:
    def __init__(self, config, testset, labelset, resultsaver, process_id):
        self.max_step = 10
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.victim = RodeXLAdapter(modelpath=config['paths']['model_checkpoint'])
        self.maxiter = config['parameters']['max_iterations']
        # self.maxiter = config['parameters']['max_iterations']
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
        flag = "selected"
        return flag
    
    def jud_type(self, vicstr:str, iteration:int) -> str:
        try:
            return self.typeset[iteration]
        except:
            sql_keywords = [
                r"(?i)\bselect\b", r"(?i)\bfrom\b", r"(?i)\bwhere\b", r"(?i)\binsert\b",
                r"(?i)\binto\b", r"(?i)\bvalues\b", r"(?i)\bupdate\b", r"(?i)\bset\b",
                r"(?i)\bdelete\b", r"(?i)\bcreate\b", r"(?i)\balter\b", r"(?i)\bdrop\b",
                r"(?i)\bjoin\b", r"(?i)\binner\b", r"(?i)\bleft\b", r"(?i)\bright\b",
                r"(?i)\bouter\b", r"(?i)\bgroup\b", r"(?i)\bby\b", r"(?i)\border\b",
                r"(?i)\bhaving\b", r"(?i)\bunion\b", r"(?i)\bexec\b"
            ]

            xss_patterns = [
                r"(?i)<script.*?>.*?</script.*?>", r"(?i)<.*?javascript:.*?>", 
                r"(?i)<.*?on\w+=.*?>", r"(?i)alert\(", r"(?i)document\.cookie", 
                r"(?i)<iframe.*?>.*?</iframe.*?>", r"(?i)<img.*?src=.*?>", 
                r"(?i)<.*?>", r"(?i)&lt;.*?&gt;"
            ]

            for pattern in sql_keywords:
                if re.search(pattern, vicstr):
                    return "sql"

            for pattern in xss_patterns:
                if re.search(pattern, vicstr):
                    return "xss"

            return "sql"
    
    def mcts(self, vic:str, viclabel:int, datatype="sql", iterations=100):
        inode = Node(0)
        vicmodel = TargetModel(self.victim, vic, self.resultsaver)
        if viclabel !=0: self.evasion_count+=1
        if datatype == "xss":
            self.xsscount+=1
            fuzzer = MonteCarloTreeSearchFuzzer(22, datatype, vic, viclabel, vicmodel, inode)   
        else:
            self.sqlcount+=1
            fuzzer = MonteCarloTreeSearchFuzzer(12, datatype, vic, viclabel, vicmodel, inode)
        
        for _ in range(iterations):
            result = fuzzer.monte_carlo_tree_search(max_step=self.max_step)

            if self.victim.get_pred(result) != viclabel:
                if datatype == "xss":
                    self.xss_succ+=1
                else:
                    self.sql_succ+=1

                if viclabel !=0:
                    self.evasion_succ+=1
                self.succ_iter.append(_*self.max_step)
                a_xss,a_sql = vicmodel.get_action_record()
                # print("XSS:")
                # print(a_xss)
                # print("SQL:")
                # print(a_sql)
                # print(vic)
                # print(result)
                # print(viclabel)
                # a = input()
                self.xss_selection.extend(a_xss)
                self.sql_selection.extend(a_sql)
                vicmodel.reset_action_record()
                return {'succ': True, 'tokens': result, 'raw_tokens': result}
            inode = fuzzer.get_best_child(inode)
            if datatype == "xss":
                self.xss_iter.append(_*self.max_step)
                fuzzer = MonteCarloTreeSearchFuzzer(22, datatype, result, viclabel, vicmodel, inode)
            else:
                self.sql_iter.append(_*self.max_step)
                fuzzer = MonteCarloTreeSearchFuzzer(12, datatype, result, viclabel, vicmodel, inode)
            vicmodel.reset_action_record()
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
        sql_asr = (self.sql_succ / self.sqlcount) * 100 if self.sqlcount > 0 else 0
        xss_asr = (self.xss_succ / self.xsscount) * 100 if self.xsscount > 0 else 0
        averageSQL = sum(self.sql_iter) / len(self.sql_iter) if len(self.sql_iter) > 0 else -1
        averageXSS = sum(self.xss_iter) / len(self.xss_iter) if len(self.xss_iter) > 0 else -1

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
        element_count = Counter(input_list)
        return dict(element_count)
        
    def result_caculate(self):
        result = {
            "ASR": 0,
            "SAQ": 0,
            "TAQ": 0,
            "EVN": 0,
            "ER": 0,
            "count":0,
            "sql": {
                "AF": {}
            },
            "xss": {
                "AF": {}
            }
        }

        total_attacks = self.sqlcount + self.xsscount
        total_success = self.sql_succ + self.xss_succ
        overall_asr = (total_success / total_attacks)  if total_attacks > 0 else 0
        result["ASR"] = overall_asr

        success_average_queries = np.mean(self.succ_iter) if self.succ_iter else 0
        result["SAQ"] = success_average_queries

        total_average_queries = (sum(self.sql_iter) + sum(self.xss_iter)) / (len(self.sql_iter) + len(self.xss_iter)) if (len(self.sql_iter) + len(self.xss_iter)) > 0 else -1
        result["TAQ"] = total_average_queries

        evasion_value = self.evasion_succ if self.evasion_succ > 0 else 0
        result["EVN"] = evasion_value

        evasion_ratio = (self.evasion_succ / total_attacks) * 100 if total_attacks > 0 else 0
        result["ER"] = evasion_ratio
        print(self.sql_choice)
        print(self.xss_choice)

        result["sql"]["AF"] = self.count_elements(self.sql_selection)
        result["xss"]["AF"] = self.count_elements(self.xss_selection)

        log = "Results Summary\n"
        log += "=" * 80 + "\n"
        log += "Overall Attack Success Rate (ASR): {:.2f}%\n".format(overall_asr)
        log += "Average Attack Queries for Successful Attacks (SAQ): {:.2f}\n".format(success_average_queries)
        log += "Average Steps Required (TAQ): {:.2f}\n".format(total_average_queries)
        log += "Evasion Count (EVN): {}\n".format(evasion_value)
        log += "Evasion Ratio (ER): {:.2f}%\n".format(evasion_ratio)
        log += "Total Attack Count: {}\n".format(self.attack_count)
        log += "=" * 80 + "\n"
        result["count"] = self.attack_count
        self.resultsaver.savelogDData(info=log)

        return result

    def exec(self):
        dataset = "PIK"
        res = {"ori_raw": [], "ori_label": [], "adv_raw": [], "adv_label": []}
        start_time = time.time()  
        for iteration, strload in tqdm(enumerate(self.testset), total=len(self.testset), position=self.processid + 1, desc=f"Process {self.processid}"):
            if time.time() - start_time >= 3600:
                print("时间超过一小时，强制中断计算...")
                metrics = self.result_caculate()
                with open(self.config['paths']['attack_result_save_path'] + dataset + "_ada{}.json".format(self.processid), "w", encoding='utf-8') as file:
                    json.dump(metrics, file, ensure_ascii=False, indent=4)

                return res

            if (iteration) % 10 == 5:
                self.getrecordMetrics()
                sqlcount = self.sqlcount if self.sqlcount != 0 else 1  # 防止除零
                xsscount = self.xsscount if self.xsscount != 0 else 1  # 防止除零

                print(f"current sql ASR = {self.sql_succ/sqlcount:.2f}")
                print(f"current xss ASR = {self.xss_succ/xsscount:.2f}")
            self.resultsaver.savelogDData(info="\nEXAMPLE " + str(iteration) + "...")
            data_a_type = self.jud_type(strload, iteration)

            _res = self.mcts(vic=strload, viclabel=self.labelset[iteration], datatype=data_a_type, iterations=self.maxiter)
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

def main(process=0):
    config = load_config("Config/adv_config_PIK.yaml")
    dataset = config['parameters']["dataname"]

    test_data = pd.read_csv(os.getcwd()+os.sep+"Data"+os.sep+dataset+os.sep+f"shuffled_data_part_{process}.csv")
    print(dataset+os.sep+f"shuffled_data_part_{process}.csv"+" has been loaded...")
    data_list = test_data['Text'].tolist()
    label_list = test_data["Label"].tolist()

    resultsaver = Results("te_single_thread", "MCTS", "PIK")
    attacker = MCTS(config, data_list, label_list, resultsaver, process)
    advdata = attacker.exec()
    
for i in range(1,11):
    main(process = i)
