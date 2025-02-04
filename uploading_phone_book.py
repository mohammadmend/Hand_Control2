import json
import csv
import pandas as pd
from parse_func import parse_csv_to_contacts, dictate, clean
import firebase_admin
from firebase_admin import credentials,  initialize_app, db

def upload(data):
    cred =credentials.Certificate("C:/Users/amend/Downloads/phonedatabase-c73cd-firebase-adminsdk-4czrm-c032bd807b.json")
    #if not firebase_admin.App:
    def_app = firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://phonedatabase-c73cd-default-rtdb.firebaseio.com'
    }, name='test2')
    ref = db.reference('restricted_access/secret_document3',app=def_app)
    ref.set(data)


def main():
    data=pd.read_csv("C:/Users/amend/Desktop/final_con.csv")
    parse= parse_csv_to_contacts(data)
    upload(dictate(parse))

if __name__=="__main__":
    main()