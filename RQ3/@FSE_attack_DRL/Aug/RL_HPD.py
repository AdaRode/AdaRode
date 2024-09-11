from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
from XLnet_Adapter_HPD import *
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
import gym
from gym import spaces
import json
import time




model_name='XLNet'

# dataset='PIK'
dataset='HPD'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, state, action, reward, next_state, done):
        states = torch.tensor([state], dtype=torch.float).to(self.device)
        actions = torch.tensor([action]).view(-1, 1).to(self.device)
        rewards = torch.tensor([reward], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor([next_state], dtype=torch.float).to(self.device)
        dones = torch.tensor([done], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = F.mse_loss(critic_1_q_values, td_target.detach())
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = F.mse_loss(critic_2_q_values, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


class AdversarialEnv(gym.Env):
    def __init__(self, victim_model, initial_state, label, attack_type, max_steps=100):
        super(AdversarialEnv, self).__init__()
        self.victim_model = victim_model
        self.initial_state = initial_state
        self.state = initial_state
        self.label = label
        self.attack_type = attack_type
        self.max_steps = max_steps
        self.current_step = 0
        if attack_type=="sql":
            self.action_space = spaces.Discrete(action_dim_sql)
        else:
            self.action_space = spaces.Discrete(action_dim_xss)
        self.observation_space = spaces.Box(low=0, high=self.action_space.n, shape=(self.max_steps,), dtype=np.int32)
        self.state_tracker = np.zeros(self.max_steps, dtype=np.int32)

    def reset(self):
        # 这里的state可不是[0,0,...,0]，而是样本字符串
        self.state = self.initial_state 
        self.current_step = 0
        self.state_tracker = np.zeros(self.max_steps, dtype=np.int32)
        return self.state_tracker

    def step(self, action, viclabel):
        if self.attack_type == "sql":
            attacker = SqlFuzzer(self.state)
            attacker.fuzz(action)
            next_state = attacker.current()
        else:
            attacker = XssFuzzer(self.state)
            attacker.fuzz(action)
            next_state = attacker.current()
            

        pred_label = self.victim_model.get_pred(next_state)[0]
        # print("pred_label",pred_label)
        reward = -1
        done = False

        if pred_label != viclabel:  # Model detects it as a normal sample
            reward = 10
            done = True
        elif self.current_step >= self.max_steps - 1:
            done = True

        self.state = next_state  # 这里的state可不是[0,0,...,0]，而是样本字符串
        self.state_tracker[self.current_step] = action  # 这里的state_tracker才是[0,0,...,0]
        self.current_step += 1

        return self.state_tracker, reward, done, {}

    def render(self, mode='human'):
        pass




def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class RS:
    def __init__(self, config, testset, labelset, typeset, resultsaver,process_id, sac_agent_sql, sac_agent_xss):
        self.config = config
        self.testset = testset
        self.labelset = labelset
        self.typeset = typeset
        self.resultsaver = resultsaver
        self.sac_agent_sql = sac_agent_sql
        self.sac_agent_xss = sac_agent_xss
        self.maxiter = config['parameters']['max_iterations']
        self.patience = config['parameters']['patience']
        self.xsscount = 1
        self.sqlcount = 1
        self.xss_succ = 0
        self.sql_succ = 0
        self.xss_iter = []
        self.sql_iter = []
        self.processid = process_id

        if model_name.startswith('DeepXSS') or model_name=='TR-IDS' or model_name=='C-BLA':
            self.victim = DeepXSSAdapter(model_name=model_name,dataset=dataset)
        if model_name.startswith('DetANN'):
            self.victim = DNNAdapter(model_name=model_name,dataset=dataset)
        if model_name.startswith('Graph'):
            self.victim = GraphXSSAdapter(model_name=model_name,dataset=dataset)
        if model_name=='RoBERTa' or model_name=='XLNet' or model_name=='Ada':
            self.victim = Sec2vecAdapter(model_name=model_name,dataset=dataset)
          
    
    
    
    def rands(self, vic:str, viclabel:str, datatype = "sql", _max_iter=100):
        if len(vic) <= 0:
            return {'succ': False, 'tokens': None, 'raw_tokens': None}

        env = AdversarialEnv(self.victim, vic, viclabel, datatype, self.maxiter)
        state = env.reset()
        episode_return = 0
        jumpup = 0
        tokens = vic
        
        if datatype == "xss":self.xsscount+=1
        if datatype == "sql":self.sqlcount+=1

        action_list=[]
        for iteration in range(1, 1+_max_iter):
            if datatype == "sql":
                action = self.sac_agent_sql.take_action(state)
                action_list.append(action)
                # print("action",action)
                # print("state",state)
                next_state, reward, done, _ = env.step(action,viclabel)
                self.sac_agent_sql.update(state, action, reward, next_state, done)
            else:
                action = self.sac_agent_xss.take_action(state)
                action_list.append(action)
                # print("action",action)
                # print("state",state)
                next_state, reward, done, _ = env.step(action,viclabel)
                self.sac_agent_xss.update(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            # print("ite",iteration,episode_return)

            if done:
                # print("episode",episode_return)
                # if episode_return>10:
                #     a=input()
                if reward > 0:
                    if datatype == "xss":
                        self.xss_succ += 1
                        self.xss_iter.append(iteration)
                    if datatype == "sql":
                        self.sql_succ += 1
                        self.sql_iter.append(iteration)
                    return {'succ': True, 'tokens': state, 'raw_tokens': state, 'action_list':action_list}
                break

        if datatype == "xss":
            self.xss_iter.append(iteration)
        if datatype == "sql":
            self.sql_iter.append(iteration)
        return {'succ': False, 'tokens': None, 'raw_tokens': None, 'action_list':action_list}
    

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
        # try:
        #     return self.typeset[iteration]
        # except:
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
        sql_success_action=[]
        xss_success_action=[]
        sql_fail_action=[]
        xss_fail_action=[]
        sql_attack_count=1
        xss_attack_count=1
        sql_attack_succ_count=0
        xss_attack_succ_count=0
        original_time = time.time()
        max_time = 3600

        for iteration, strload in tqdm(enumerate(self.testset), total=len(self.testset),position=self.processid + 1,desc=f"Process {self.processid}"):
            # if self.victim.get_pred(strload)[0]!=1:
            #     continue
            if time.time() - original_time > max_time:
                break
            if (iteration) % 10 == 5:
                self.getrecordMetrics()
                print(f"current sql ASR = {self.sql_succ/self.sqlcount:.2f}")
                print(f"current xss ASR = {self.xss_succ/self.xsscount:.2f}")
                with open("/root/autodl-tmp/@ICSE_RQ3_RL/Aug/Augmentation/augdata/HPD/"+model_name+'.pickle', "wb") as file:
                    pickle.dump(res, file)
            # print("\nEXAMPLE " + str(iteration) + "...")
            self.resultsaver.savelogDData(info="\nEXAMPLE " + str(iteration) + "...")
            start_time = time.time()
            # print("typeset[iteration]",self.typeset[iteration])
            data_a_type = self.jud_type(strload, iteration)
            # print(data_a_type)
            # print(self.victim.get_pred(strload)[0],data_a_type)
            # if data_a_type!="sql":
            #     continue
            # self.maxiter = 5
            # print(self.victim.get_pred(strload)[0],data_a_type)
            _res = self.rands(vic=strload, viclabel=self.labelset[iteration],
                            datatype=data_a_type, _max_iter=self.maxiter)
            
            if self.labelset[iteration]!=0:
                if self.labelset[iteration]==1:
                    sql_attack_count+=1
                    if _res['succ']:
                        sql_attack_succ_count+=1
                else:
                    xss_attack_count+=1
                    if _res['succ']:
                        xss_attack_succ_count+=1
                
            if _res['succ']:
                # Bypass successfully
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " SUCCEEDED!")
                timecost = "time cost = %.2f min" % ((time.time() - start_time) / 60)
                self.resultsaver.savelogDData(info=timecost)
                res['adv_label'].append(self.labelset[iteration])
                res['adv_raw'].append(_res['raw_tokens'])
                res["ori_raw"].append(strload)
                res["ori_label"].append(self.labelset[iteration])
                if data_a_type=='sql':
                    sql_success_action.append(_res['action_list'])
                else:
                    xss_success_action.append(_res['action_list'])
            else:
                # print("EXAMPLE " + str(iteration) + " FAILED...")
                self.resultsaver.savelogDData(info="EXAMPLE " + str(iteration) + " FAILED...")
                res['adv_raw'].append(strload)
                res['adv_label'].append(self.labelset[iteration])
                res["ori_raw"].append(strload)
                res["ori_label"].append(self.labelset[iteration])
                if data_a_type=='sql':
                    sql_fail_action.append(_res['action_list'])
                else:
                    xss_fail_action.append(_res['action_list'])
        
        result=dict()
        result['sql']=dict()
        result['xss']=dict()
        SAQ_sql=0
        SAQ_xss=0
        TAQ_sql=0
        TAQ_xss=0
        element_count_sql={}
        element_count_xss={}
        for sa in sql_success_action:
            SAQ_sql+=len(sa)
        for sa in xss_success_action:
            SAQ_xss+=len(sa)
        for a in sql_success_action+sql_fail_action:
            TAQ_sql+=len(a)
        for a in xss_success_action+xss_fail_action:
            TAQ_xss+=len(a)
        for sa in sql_success_action:
            # 遍历每个子列表中的元素
            for element in sa:
                # 如果元素在字典中已经存在,则计数加1
                if element in element_count_sql:
                    element_count_sql[element] += 1
                # 如果元素不存在,则添加到字典中并设置计数为1
                else:
                    element_count_sql[element] = 1
        for sa in xss_success_action:
            # 遍历每个子列表中的元素
            for element in sa:
                # 如果元素在字典中已经存在,则计数加1
                if element in element_count_xss:
                    element_count_xss[element] += 1
                # 如果元素不存在,则添加到字典中并设置计数为1
                else:
                    element_count_xss[element] = 1
                # 设置最大执行时间为60秒(1分钟)
                
        result['sql']['ASR']=self.sql_succ/self.sqlcount
        result['sql']['SAQ']=SAQ_sql/max(1,len(sql_success_action))
        result['sql']['TAQ']=TAQ_sql/max(1,len(sql_success_action+sql_fail_action))
        result['sql']['EVN']=sql_attack_succ_count
        result['sql']['ER']=sql_attack_succ_count/max(1,sql_attack_count)
        result['sql']['AF']=element_count_sql
        result['sql']['Count']=self.sqlcount
        result['xss']['ASR']=self.xss_succ/self.xsscount
        result['xss']['SAQ']=SAQ_xss/max(1,len(xss_success_action))
        result['xss']['TAQ']=TAQ_xss/max(1,len(xss_success_action+xss_fail_action))
        result['xss']['EVN']=xss_attack_succ_count
        result['xss']['ER']=xss_attack_succ_count/max(1,xss_attack_count)
        result['xss']['AF']=element_count_xss
        result['xss']['Count']=self.xsscount

        result['ASR']=(self.sql_succ+self.xss_succ)/(self.sqlcount+self.xsscount)
        result['SAQ']=(SAQ_sql+SAQ_xss)/max(1,len(sql_success_action)+len(xss_success_action))
        result['TAQ']=(TAQ_sql+TAQ_xss)/max(1,len(sql_success_action+sql_fail_action)+len(xss_success_action+xss_fail_action))
        result['EVN']=sql_attack_succ_count+xss_attack_succ_count
        result['ER']=(sql_attack_succ_count+xss_attack_succ_count)/max(1,sql_attack_count+xss_attack_count)
        result['sql']['AF']=element_count_sql
        result['xss']['AF']=element_count_xss
        result['count']=self.xsscount+self.sqlcount
        
        
        
        with open("/root/autodl-tmp/@ICSE_RQ3_RL/Aug/Augmentation/augdata/HPD/"+model_name+config['paths'][dataset]['data'].split('.')[0].replace('/','_')+".json", "w") as f:
            json.dump(result, f)
        
        self.getrecordMetrics()
        
        return res


def run_process(process_id, data_slice, label_slice, type_slice, config, sac_agent_sql, sac_agent_xss):
    device = torch.device(config['parameters']['device'])
    torch.cuda.set_device(device)
    
    resultsaver = Results(f"te{len(data_slice)}_{process_id}_AdaA", "RandomSample", "SIK")
    attacker = RS(config, data_slice, label_slice, type_slice, resultsaver, process_id, sac_agent_sql, sac_agent_xss)
    advdata = attacker.exec()
    return advdata
    
    
if __name__ == "__main__":
    # 加载配置文件
    config = load_config("Config/adv_config_HPD.yaml")

    # 读取数据
    test_data = pd.read_csv(config['paths'][dataset]['data'])

    # 获取所有数据
    data_list = test_data['Text'].tolist()
    label_list = test_data["Label"].tolist()
    type_list = test_data["type"].tolist()

    # 初始化SAC智能体
    state_dim = config['parameters']['max_iterations']
    action_dim_sql = 12
    action_dim_xss = 21
    hidden_dim = 256
    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 1e-2
    gamma = 0.98
    tau = 0.005
    target_entropy = -1
    device = torch.device(config['parameters']['device'])

    sac_agent_sql = SAC(state_dim, hidden_dim, action_dim_sql, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    sac_agent_xss = SAC(state_dim, hidden_dim, action_dim_xss, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    # 直接运行处理函数，传递所有数据
    advdata = run_process(0, data_list, label_list, type_list, config, sac_agent_sql, sac_agent_xss)

    # # 保存合并后的数据
    # with open(config['paths'][dataset]['augmented_data_save_path']+model_name+'__.pickle', "wb") as file:
    #     pickle.dump(advdata, file)

    # print(config['paths'][dataset]['augmented_data_save_path']+model_name+'__.pickle')