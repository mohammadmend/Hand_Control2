import os
from twilio.rest import Client

def maincall(number):
    account_sid = "ACf36773da16bc93514a7df0ceab365470"
    auth_token = "4d99a0eea34f29c9c555fe08c34bae6c"
    client = Client(account_sid, auth_token)

    call = client.calls.create(
    to=("+",number),
    from_="+18557293193"
    )
    print(("+",number))
    print(call.sid)

if __name__=="__maincall__":
    maincall()