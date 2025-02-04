import firebase_admin
from firebase_admin import credentials,  initialize_app, db
from openai import OpenAI
from call_twilio import maincall
OpenAI.api_key = "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"



def init_fi():
    cred=credentials.Certificate("C:/Users/amend/Downloads/phonedatabase-c73cd-firebase-adminsdk-4czrm-c032bd807b.json")
    def_app=firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://phonedatabase-c73cd-default-rtdb.firebaseio.com'
    })
    return db.reference('restricted_access/secret_document3')

def retriveKey(ref):
    
    data=ref.get()
    return list(data.keys())


def retrieveContact(name_list, transcription):
    client1 = OpenAI(api_key= "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")
    # client1.api_key= "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"

    completion = client1.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content":"You are given a transcription of a name and a list of names. Find the most similar name in the list."},
        {
            "role": "user",
            "content": f"Find the most similar name to '{transcription}' from this list: {name_list}, response should just be the name found nothing else ignore '-' in between letters ! "
        }
    ]
)

    response =(completion.choices[0].message)

    return response
def retrieveFullContact(resposne,ref):
    data=ref.child(resposne).get()
    return data

def main():
    ref=init_fi()
    list=(retriveKey(ref))
    name=retrieveContact(list,"-D-a-d-")
    print(name.content)
    info=retrieveFullContact(name.content,ref)
    print(info['telephone'])
    #maincall(info['telephone'])
if __name__=="__main__":
    main()
    