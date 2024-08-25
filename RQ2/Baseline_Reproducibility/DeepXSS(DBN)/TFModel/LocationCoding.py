
import pickle
import os
from tqdm import tqdm



# 传入对应的A处理的数据结构，返回Tokenization.pickle文件
class LocCoding:
    def __init__(self,codeset, dataname, savepath):
        self.codeset = codeset
        self.dataname = dataname
        self.savepath = savepath
    
    # 分割字符串用的(内部函数)
    def _splitCharacters(self,str_to_split):
        #Character_sets = ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',']
        str_list_str = ''
        
        if '(' in str_to_split:
            str_to_split = str_to_split.replace('(', ' ( ') # Add the space before and after the '(', so that it can be split by space.
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if ')' in str_to_split:
            str_to_split = str_to_split.replace(')', ' ) ') # Add the space before and after the ')', so that it can be split by space.
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '{' in str_to_split:
            str_to_split = str_to_split.replace('{', ' { ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '}' in str_to_split:
            str_to_split = str_to_split.replace('}', ' } ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '*' in str_to_split:
            str_to_split = str_to_split.replace('*', ' * ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '/' in str_to_split:
            str_to_split = str_to_split.replace('/', ' / ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '+' in str_to_split:
            str_to_split = str_to_split.replace('+', ' + ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '-' in str_to_split:
            str_to_split = str_to_split.replace('-', ' - ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '=' in str_to_split:
            str_to_split = str_to_split.replace('=', ' = ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if ';' in str_to_split:
            str_to_split = str_to_split.replace(';', ' ; ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '[' in str_to_split:
            str_to_split = str_to_split.replace('[', ' [ ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if ']' in str_to_split:
            str_to_split = str_to_split.replace(']', ' ] ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '>' in str_to_split:
            str_to_split = str_to_split.replace('>', ' > ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '<' in str_to_split:
            str_to_split = str_to_split.replace('<', ' < ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '"' in str_to_split:
            str_to_split = str_to_split.replace('"', ' " ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if '->' in str_to_split:
            str_to_split = str_to_split.replace('->', ' -> ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '>>' in str_to_split:
            str_to_split = str_to_split.replace('>>', ' >> ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if '<<' in str_to_split:
            str_to_split = str_to_split.replace('<<', ' << ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
        
        if ',' in str_to_split:
            str_to_split = str_to_split.replace(',', ' , ')
            str_list = str_to_split.split(' ')
            str_list_str = ' '.join(str_list)
            
        if str_list_str != '':
            return str_list_str
        else:
            return str_to_split

    # 处理对应的字符串输入为[[str1],[str2],....]
    # 处理对应的字符串输出为[[char1，char2],[char1，char2],....]
    def dataloader(self, codelist: list) -> list:
        files_list = []
        for code in tqdm(codelist, desc="Processing codes", unit="code"):
            lines = code.split("\n")
            file_list = [] 
            for line in lines:
                sub_line = line.split()
                new_sub_line = []
                for element in sub_line:
                    new_element = self._splitCharacters(element)
                    new_sub_line.append(new_element)
                new_line = ' '.join(new_sub_line)
                file_list.append(new_line)
            new_file_list = ' '.join(file_list)
            split_by_space = new_file_list.split()
            files_list.append(split_by_space)
        return files_list

    def getCodesMatrix(self):
        return self.dataloader(self.codeset)

    # 编码后存储
    def Tokenization(self):
        from keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=None, filters=',', lower=False, char_level=False, oov_token=None)
        data_list = self.dataloader(self.codeset)
        tokenizer.fit_on_texts(data_list)
        # Save the tokenizer.
        tmp_path = self.savepath + os.sep + self.dataname + os.sep
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        with open(tmp_path+'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
        print("-"*80)
        print("Tokenization completed!")


    