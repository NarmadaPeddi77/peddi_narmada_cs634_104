#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations

class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

def insert_tree(transaction, tree, header_table, count):
    if transaction[0] in tree.children:
        tree.children[transaction[0]].count += count
    else:
        tree.children[transaction[0]] = TreeNode(transaction[0], count, tree)
        if header_table[transaction[0]][1] is None:
            header_table[transaction[0]][1] = tree.children[transaction[0]]
        else:
            update_header(header_table[transaction[0]][1], tree.children[transaction[0]])
    if len(transaction) > 1:
        insert_tree(transaction[1:], tree.children[transaction[0]], header_table, count)

def update_header(node, target_node):
    while node.link is not None:
        node = node.link
    node.link = target_node

def construct_fp_tree(data, min_support):
    item_counts = {}
    for transaction, count in data.items():
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + count
    item_counts = {k: v for k, v in item_counts.items() if v >= min_support}
    header_table = {item: [count, None] for item, count in item_counts.items()}
    tree_root = TreeNode(None, 1, None)
    for transaction, count in data.items():
        sorted_items = [item for item in transaction if item in item_counts]
        sorted_items.sort(key=lambda x: item_counts[x], reverse=True)
        if len(sorted_items) > 0:
            insert_tree(sorted_items, tree_root, header_table, count)
    return tree_root, header_table

def mine_tree(header_table, min_support, prefix, frequent_itemsets):
    sorted_items = sorted(list(header_table.items()), key=lambda p: p[1][0])
    for item, (count, node) in sorted_items:
        new_prefix = prefix.copy()
        new_prefix.add(item)
        frequent_itemsets[tuple(sorted(new_prefix))] = count
        conditional_pattern_base = find_conditional_pattern_base(node)
        conditional_tree_data = {}
        for pattern, count in conditional_pattern_base:
            conditional_tree_data[pattern] = count
        conditional_tree_root, conditional_header_table = construct_fp_tree(conditional_tree_data, min_support)
        if conditional_header_table:
            mine_tree(conditional_header_table, min_support, new_prefix, frequent_itemsets)

def find_conditional_pattern_base(node):
    patterns = []
    while node is not None:
        prefix_path = ascend_tree(node)
        if len(prefix_path) > 1:
            patterns.append((tuple(prefix_path[1:]), node.count))
        node = node.link
    return patterns

def ascend_tree(node):
    path = []
    while node and node.parent:
        path.append(node.item)
        node = node.parent
    return path

def fp_growth(data, min_support):
    tree, header_table = construct_fp_tree(data, min_support)
    frequent_itemsets = {}
    mine_tree(header_table, min_support, set(), frequent_itemsets)
    return frequent_itemsets

def convert_to_freq_dict(transactions):
    freq_dict = {}
    for transaction in transactions:
        transaction_tuple = tuple(sorted(transaction))
        freq_dict[transaction_tuple] = freq_dict.get(transaction_tuple, 0) + 1
    return freq_dict

def read_transactions_from_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None

def load_transactions(dataList):
    Transactions = []
    df_items = dataList['TransactionList']
    comma_splitted_df = df_items.apply(lambda x: x.split(','))
    for i in comma_splitted_df:
        Transactions.append(i)
    return Transactions

def generate_association_rules(frequent_itemsets, min_confidence):
    association_rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            rules_from_itemset = generate_rules_from_itemset(itemset, frequent_itemsets, min_confidence)
            association_rules.extend(rules_from_itemset)
    return association_rules

def generate_rules_from_itemset(itemset, frequent_itemsets, min_confidence):
    rules = []
    subsets = get_subsets(itemset)
    for subset in subsets:
        remaining = tuple(set(itemset).difference(subset))
        confidence = frequent_itemsets[itemset] / frequent_itemsets[subset]
        if confidence >= min_confidence:
            rules.append((subset, remaining, confidence))
    return rules

def get_subsets(itemset):
    subsets = []
    for i in range(1, len(itemset)):
        subsets.extend(combinations(itemset, i))
    return subsets

if __name__ == '__main__':
    print("Welcome to the FP-Growth algorithm.")
    print("Please choose the dataset you want:")
    print("1. Nike")
    print("2. Kmart")
    print("3. Vehicles")
    print("4. Sports")
    print("5. Retail")

    while True:
        choice_of_data = input("Enter your choice: ")
        file_path = ""
        if choice_of_data == '1':
            file_path = 'Nike.csv'
            print('User chose Nike dataset')
            break
        elif choice_of_data == '2':
            file_path = ('Kmart - Sheet1.csv')
            print('User chose Grocery Store dataset')
            break
        elif choice_of_data == '3':
            file_path = 'Cars_List.csv'
            print('User chose Cars List dataset')
            break
        elif choice_of_data == '4':
            file_path = 'Games_Transaction_List.csv'
            print('User chose Games dataset')
            break
        elif choice_of_data == '5':
            file_path = 'Costco.csv'
            print('User chose Costco dataset')
            break
        else:
            print("Invalid choice. Please enter the number corresponding to the dataset.")

    dataList = read_transactions_from_csv(file_path)
    if dataList is None:
        sys.exit(1)

    minsupport = float(input("Enter the Minimum Support (in percentage): ")) / 100
    minconfidence = float(input("Enter the Minimum Confidence (in percentage): ")) / 100

    Transactions = load_transactions(dataList)
    data = convert_to_freq_dict(Transactions)

    start_time = time.time()
    frequent_itemsets = fp_growth(data, minsupport)
    association_rules = generate_association_rules(frequent_itemsets, minconfidence)
    end_time = time.time()

    print("\nFrequent itemsets found with FP-Growth algorithm:")
    for itemset, support in frequent_itemsets.items():
        print(f"Itemset: {itemset}, Support: {support}")

    print("\nAssociation rules found with FP-Growth algorithm:")
    for rule in association_rules:
        antecedent = ', '.join(rule[0])
        consequent = ', '.join(rule[1])
        confidence = rule[2]  # Accessing confidence from the rule tuple
        print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
        
    print("-------------------------- RUNNING TIME:------------------------------------")
    print("The Runtime of the program is: " + str(end_time - start_time) + "seconds")
    
    print("---------------------------------------------------------------------------\n")

       
              
import sys
import time
import numpy as np
import pandas as pd

print("Welcome to the apriori algorithms. \n Please chose the dataset you want: \n 1. Nike \n 2.Kmart \n 3.Vehicles \n 4. Sports \n 5.Costco") 
while True:
    choice_of_data=input()
    if(choice_of_data=='1'):
        dataList=pd.read_csv('Nike.csv')
        print('User chose Test dataset')
        break
    elif(choice_of_data=='2'):
        dataList=pd.read_csv('Kmart - Sheet1.csv')
        print('User chose Kmart dataset')
        break
    elif(choice_of_data=='3'):
        dataList=pd.read_csv('Cars_List.csv')
        print('User chose Cars  dataset')
        break
    elif(choice_of_data=='4'):
        dataList=pd.read_csv('Games_Transaction_List.csv')
        print('User chose Sports dataset')
        break
    elif(choice_of_data=='5'):
        dataList=pd.read_csv('Costco.csv')
        print('User chose Costco dataset')
        break
    else:
        print("Invalid data, please enter the number corresponding to the data")
        break     
    


print("Enter the Minimum Support (in percentage) : ", end=" ")
minsupport = input()
print("Enter the Minimum Confidence (in percentage) : ", end=" ")
minconfidence = input()
min_support = float(minsupport)/100
min_conf = float(minconfidence)/100
print('\n')
print("The minimum support is :", minsupport)
print("The minimum Confidence is :",minconfidence)

def load_transactions(dataList):
    Transactions = []
    df_items = dataList['TransactionList']
    comma_splitted_df = df_items.apply(lambda x: x.split(','))
    for i in comma_splitted_df:
        Transactions.append(i)
    return Transactions
load_transactions(dataList)

Transactions = load_transactions(dataList)
Transactions

class Rule:

    def __init__(self, left, right, all):
        self.left = list(left)
        self.left.sort()
        self.right = list(right)
        self.right.sort()
        self.all = all

    def __str__(self):
        return ",".join(self.left)+" ==> "+",".join(self.right)

    def __hash__(self):
        return hash(str(self))
def scan(Transactions, Ck):
    count = {s: 0 for s in Ck}
    n = len(Transactions)
    for t in Transactions:
        for fset in Ck:
            if fset.issubset(t):
                count[fset] += 1
    
    return {fset: support/n for fset, support in count.items() if support/n>=min_support}
def calculateCandidate(Lk):
    res = []
    print("len Lk", len(Lk))
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):
            it1 = Lk[i]
            it2 = Lk[j]
            it11 = list(it1)
            it22 = list(it2)
            it11.sort()
            it22.sort()
            print("it11",it11)
            print("it22",it22)
            if it11[:len(it1)-1] == it22[:len(it1)-1]:
                res.append(it1 | it2)
                print("Res: ", res)
    return res
def calculate_frequency_support():
    support = {}
    candidate = [[]]
    Lk = [[]]
    C1 = set()
    for t in Transactions:
        for item in t:
            C1.add(frozenset([item]))
            #print(C1)
            #print("*****")
    print("----------------------------------------")
    print("C1: ", C1)
    candidate.append(C1)
    #print(candidate)
    print("----------------------------------------")
    print("Transactions: ",Transactions)
    count = scan(Transactions, C1)
    print("----------------------------------------")
    print("Count: ", count)
    Lk.append(list(count.keys()))
    print("----------------------------------------")
    print("Lk: ", Lk)
    support.update(count)
    print("----------------------------------------")
    print("support: ", support)
    print("----------------------------------------")
    print("candidate: ",candidate)
    k = 1
    while len(Lk[k]) > 0:
        print("+++++++++++++++++++++++++++++++++++++++")
        print("k=", k)
        print("Lk[k]: ", Lk[k])
        candidate.append(calculateCandidate(Lk[k]))
        print("candidate: ", candidate)
        print("candidate[k+1]: ",candidate[k+1])
        count = scan(Transactions, candidate[k+1])
        support.update(count)
        Lk.append(list(count.keys()))
        k += 1
    return Lk, support
def EvaluateSecondaryRules(fs, rights, fresult, support):
    rlength = len(rights[0])
    totlength = len(fs)
    if totlength-rlength > 0:
        rights = calculateCandidate(rights)
        new_right = []
        for right in rights:
            left = fs - right
            if len(left) == 0:
                continue
            confidence = support[fs] / support[left]
            if confidence >= min_conf:
                fresult.append([Rule(left, right, fs), support[fs],  confidence])
                new_right.append(right)

        if len(new_right) > 1:
            EvaluateSecondaryRules(fs, new_right, fresult, support)
def EvaluateAssociationRules(frequent, support):
    fresult = []
    for i in range(2, len(frequent)):
        if len(frequent[i]) == 0:
            break
        freq_sets = frequent[i]

        for fs in freq_sets:
            for right in [frozenset([x]) for x in fs]:
                left = fs-right
                confidence = support[fs] / support[left]
                if confidence >= min_conf:
                    fresult.append([Rule(left, right, fs), support[fs], confidence])

        if len(freq_sets[0]) != 2:

            for fs in freq_sets:
                right = [frozenset([x]) for x in fs]
                EvaluateSecondaryRules(fs, right, fresult, support)

    fresult.sort(key=lambda x: str(x[0]))
    return fresult
if __name__ == '__main__':

    start_time = time.time()
    freq, supp = calculate_frequency_support()
    print("Frequency: ",freq)
    print("Support: ", supp)
    fresult = EvaluateAssociationRules(freq, supp)
    end_time = time.time()
    
    print("\n----- > Association With Support and Confidence: < -------\n")
    for x in fresult:
        print("Rule: ",x[0])
        print("Support: ", x[1])
        print("Confidence: ", x[2])
        print("\n")
        
    print("-------------------------- RUNNING TIME:------------------------------------")
    print("The Runtime of the program is: " + str(end_time - start_time) + "seconds")
    
    print("---------------------------------------------------------------------------\n")
    
    
import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations

def convert_to_freq_dict(transactions):
    freq_dict = {}
    for transaction in transactions:
        transaction_tuple = tuple(sorted(transaction))
        freq_dict[transaction_tuple] = freq_dict.get(transaction_tuple, 0) + 1
    return freq_dict

def read_transactions_from_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None

def load_transactions(dataList):
    Transactions = []
    df_items = dataList['TransactionList']
    comma_splitted_df = df_items.apply(lambda x: x.split(','))
    for i in comma_splitted_df:
        Transactions.append(i)
    return Transactions

def generate_frequent_itemsets_brute_force(transactions, min_support):
    itemsets = set()
    for transaction in transactions:
        for item in transaction:
            itemsets.add(frozenset([item]))

    frequent_itemsets = {}
    total_transactions = len(transactions)
    for itemset in itemsets:
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        support = count / total_transactions
        if support >= min_support:
            frequent_itemsets[itemset] = support

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    association_rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            rules_from_itemset = generate_rules_from_itemset(itemset, frequent_itemsets, min_confidence)
            association_rules.extend(rules_from_itemset)
    return association_rules

def generate_rules_from_itemset(itemset, frequent_itemsets, min_confidence):
    rules = []
    subsets = get_subsets(itemset)
    for subset in subsets:
        remaining = tuple(set(itemset).difference(subset))
        
        if len(subset) > 0 and len(remaining) > 0:
            subset_support = frequent_itemsets.get(subset, 0)
            if subset_support == 0:
                continue
            
            confidence = frequent_itemsets[itemset] / subset_support
            
            if confidence >= min_confidence:
                rules.append((subset, remaining, confidence))
    
    return rules

def get_subsets(itemset):
    subsets = []
    for i in range(1, len(itemset)):
        subsets.extend(combinations(itemset, i))
    return subsets

if __name__ == '__main__':
    print("Welcome to the Brute Force Association Rule Mining Algorithm.")
    
    # Data Selection
    print("Please choose the dataset you want:")
    
    datasets = {
        '1': 'Nike.csv',
        '2': 'Kmart - Sheet1.csv',
        '3': 'Cars_List.csv',
        '4': 'Games_Transaction_List.csv',
        '5': 'Costco.csv'
    }
    
    while True:
        choice_of_data = input("Enter your choice: ")
        
        if choice_of_data in datasets:
            file_path = datasets[choice_of_data]
            print(f"User chose {file_path} dataset")
            break
        else:
            print("Invalid choice. Please enter the number corresponding to the dataset.")
    
    # Minimum Support and Confidence Input
    minsupport = float(input("Enter the Minimum Support (in percentage): ")) / 100
    minconfidence = float(input("Enter the Minimum Confidence (in percentage): ")) / 100
    
    # Data Processing and Analysis
    dataList = read_transactions_from_csv(file_path)
    
    if dataList is None:
        sys.exit(1)
    
    Transactions = load_transactions(dataList)
    
    start_time = time.time()
    
    frequent_itemsets = generate_frequent_itemsets_brute_force(Transactions, minsupport)
    
    association_rules = generate_association_rules(frequent_itemsets, minconfidence)
    
    # Output Display
    print("\nFrequent itemsets found with Brute Force algorithm:")
    for itemset, support in frequent_itemsets.items():
        print(f"Itemset: {', '.join(itemset)}, Support: {support}")
     
    print("\nAssociation rules found with Brute Force algorithm:")
    for rule in association_rules:
        antecedent = ', '.join(rule[0])
        consequent = ', '.join(rule[1])
        confidence = rule[2]
        print(f"Rule: {{{antecedent}}} -> {{{consequent}}}, Confidence: {confidence}")
    
    # Runtime Display   
    end_time = time.time()
    print("-------------------------- RUNNING TIME:------------------------------------")
    print("The Runtime of the program is: " + str(end_time - start_time) + "seconds")
    print("---------------------------------------------------------------------------\n")

