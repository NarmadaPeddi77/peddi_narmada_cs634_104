#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

       

