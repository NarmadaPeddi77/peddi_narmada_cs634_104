#!/usr/bin/env python
# coding: utf-8

# In[25]:


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

