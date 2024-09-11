import os
import json
import math
import numpy as np

directories = ['./RS', './DRL', './AdvSQLi(MCTS)', './AdaRode']

def calculate_cv(af):
    values = list(af.values())
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    stddev = math.sqrt(variance)
    return stddev / mean if mean != 0 else 0

for directory_path in directories:
    asr_list = []
    saq_list = []
    taq_list = []
    evn_list = []
    er_list = []
    count_list = []
    cv_list = []
    sample_list = []
    quality_list = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                asr = data.get('ASR', 0)
                if asr <= 1:
                    asr *= 100  
                asr_list.append(asr)
                
                saq_list.append(data.get('SAQ', 0))
                taq_list.append(data.get('TAQ', 0))
                evn_list.append(data.get('EVN', 0))
                er_list.append(data.get('ER', 0))
                count_list.append(data.get('count', 0))
                if data.get('ASR', 0) > 1: AdaSR = data.get('ASR', 0)/100
                else: AdaSR = data.get('ASR',0)
                sample_list.append(AdaSR*data.get('count', 0))
                
                if 'sql' in data:
                    af_sql = data['sql'].get('AF', {})
                    cv_sql = calculate_cv(af_sql)
                    if cv_sql is not None:
                        cv_list.append(cv_sql)

                if 'xss' in data:
                    af_xss = data['xss'].get('AF', {})
                    cv_xss = calculate_cv(af_xss)
                    if cv_xss is not None:
                        cv_list.append(cv_xss)

                if cv_list:
                    quality = (AdaSR * data.get('count', 0)) / np.mean(cv_list)
                    quality_list.append(quality)
    
    if asr_list:
        mean_asr = np.mean(asr_list)
        median_asr = np.median(asr_list)
        max_asr = np.max(asr_list)
        min_asr = np.min(asr_list)
        std_asr = np.std(asr_list)
        var_asr = np.var(asr_list)
        cv_asr = std_asr / mean_asr if mean_asr != 0 else 0

        mean_saq = np.mean(saq_list)
        median_saq = np.median(saq_list)
        max_saq = np.max(saq_list)
        min_saq = np.min(saq_list)
        std_saq = np.std(saq_list)
        var_saq = np.var(saq_list)
        cv_saq = std_saq / mean_saq if mean_saq != 0 else 0

        mean_taq = np.mean(taq_list)
        median_taq = np.median(taq_list)
        max_taq = np.max(taq_list)
        min_taq = np.min(taq_list)
        std_taq = np.std(taq_list)
        var_taq = np.var(taq_list)
        cv_taq = std_taq / mean_taq if mean_taq != 0 else 0

        mean_evn = np.mean(evn_list)
        median_evn = np.median(evn_list)
        max_evn = np.max(evn_list)
        min_evn = np.min(evn_list)
        std_evn = np.std(evn_list)
        var_evn = np.var(evn_list)
        cv_evn = std_evn / mean_evn if mean_evn != 0 else 0

        mean_er = np.mean(er_list)
        median_er = np.median(er_list)
        max_er = np.max(er_list)
        min_er = np.min(er_list)
        std_er = np.std(er_list)
        var_er = np.var(er_list)
        cv_er = std_er / mean_er if mean_er != 0 else 0

        mean_count = np.mean(count_list)
        median_count = np.median(count_list)
        max_count = np.max(count_list)
        min_count = np.min(count_list)
        std_count = np.std(count_list)
        var_count = np.var(count_list)
        cv_count = std_count / mean_count if mean_count != 0 else 0

        mean_sample = np.mean(sample_list)
        median_sample = np.median(sample_list)
        max_sample = np.max(sample_list)
        min_sample = np.min(sample_list)
        std_sample = np.std(sample_list)
        var_sample = np.var(sample_list)
        cv_sample = std_sample / mean_sample if mean_sample != 0 else 0

        if cv_list:
            mean_cv = np.mean(cv_list)
            median_cv = np.median(cv_list)
            max_cv = np.max(cv_list)
            min_cv = np.min(cv_list)
            std_cv = np.std(cv_list)
            var_cv = np.var(cv_list)
            cv_cv = std_cv / mean_cv if mean_cv != 0 else 0
        else:
            mean_cv = median_cv = max_cv = min_cv = std_cv = var_cv = cv_cv = None

        if quality_list:
            mean_quality = np.mean(quality_list)
            median_quality = np.median(quality_list)
            max_quality = np.max(quality_list)
            min_quality = np.min(quality_list)
            std_quality = np.std(quality_list)
            var_quality = np.var(quality_list)
            cv_quality = std_quality / mean_quality if mean_quality != 0 else 0
        else:
            mean_quality = median_quality = max_quality = min_quality = std_quality = var_quality = cv_quality = None

        print(f"Directory: {directory_path}")
        print(f"Mean R_as: {mean_asr}, Median R_as: {median_asr}, Max R_as: {max_asr}, Min R_as: {min_asr}, Std R_as: {std_asr}, Var R_as: {var_asr}, CV R_as: {cv_asr}")
        print("================================================================")
        print(f"Mean R_esc: {mean_er}, Median R_esc: {median_er}, Max R_esc: {max_er}, Min R_esc: {min_er}, Std R_esc: {std_er}, Var R_esc: {var_er}, CV R_esc: {cv_er}")
        print("================================================================")
        print(f"Mean I_sa: {mean_saq}, Median I_sa: {median_saq}, Max I_sa: {max_saq}, Min I_sa: {min_saq}, Std I_sa: {std_saq}, Var I_sa: {var_saq}, CV I_sa: {cv_saq}")
        print("================================================================")
        print(f"Mean I_ta: {mean_taq}, Median I_ta: {median_taq}, Max I_ta: {max_taq}, Min I_ta: {min_taq}, Std I_ta: {std_taq}, Var I_ta: {var_taq}, CV I_ta: {cv_taq}")
        print("================================================================")
        print(f"Mean N_succ: {mean_sample}, Median N_succ: {median_sample}, Max N_succ: {max_sample}, Min N_succ: {min_sample}, Std N_succ: {std_sample}, Var N_succ: {var_sample}, CV N_succ: {cv_sample}")
        print("================================================================")
        print(f"Mean D_succ: {mean_cv}, Median D_succ: {median_cv}, Max D_succ: {max_cv}, Min D_succ: {min_cv}, Std D_succ: {std_cv}, Var D_succ: {var_cv}, CV D_succ: {cv_cv}")
        print("================================================================")
        print("****************************************************************")
    else:

        print(f"No JSON files found in the directory {directory_path}.")
