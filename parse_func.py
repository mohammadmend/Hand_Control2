import json
import csv
import pandas as pd
import re


def parse_csv_to_contacts(df):
    contacts = []
    contact = {}
    for index, row in df.iterrows():
        if row.iloc[0] == 'N':  
            contact['last_name'] = row.iloc[1] if pd.notna(row.iloc[1]) else ''
            contact['first_name'] = row.iloc[2] if pd.notna(row.iloc[2]) else ''
        elif row.iloc[0] == 'FN':  
            contact['full_name'] = row.iloc[1]  if pd.notna(row.iloc[1]) else ''
        elif row.iloc[0] == 'TEL':  
            contact['telephone'] = row.iloc[4]  if pd.notna(row.iloc[4]) else ''
        elif row.iloc[0] == 'END':  
            contacts.append(contact)
            contact = {} 
    
    return contacts

def clean(key):
    return re.sub(r'[^\w]', '_', key.strip())

def dictate(lis):
    data={clean(person['first_name']):person for person in lis}
    return {key: value for key, value in data.items() if key != ''}
    
def main():
    df=pd.read_csv("C:/Users/amend/Desktop/final_con.csv")
    c=parse_csv_to_contacts(df)
    print((dictate(c)))


    
if __name__=="__main__":
    main()