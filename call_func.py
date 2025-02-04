import json
import csv
import pandas as pd

def parse_csv_to_contacts(df):
    contacts = []
    contact = {}
    
   
    for index, row in df.iterrows():
        if row[0] == 'N':  # Name row
            contact['last_name'] = row[1]  # Last name
            contact['first_name'] = row[2]  # First name
        elif row[0] == 'FN':  # Full name row
            contact['full_name'] = row[1]  # Full name
        elif row[0] == 'TEL':  # Telephone row
            contact['telephone'] = row[4]  # Telephone number (from 4th column)
        elif row[0] == 'END':  # End of a contact entry
            contacts.append(contact)  # Add contact to the list
            contact = {}  # Reset contact for the next entry
    
    return contacts




    
def main():
    df=pd.read_csv("C:/Users/amend/Desktop/final_con.csv")
    c=parse_csv_to_contacts(df)
    
if __name__=="__main__":
    main()