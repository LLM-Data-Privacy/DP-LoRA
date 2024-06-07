import requests
import time

server_url = "http://flask-server:3000"
actions = ["get","train","upload","fuck"]
def enter(i):
    return actions[i%4]
    
def get_broadcast():
    resp = requests.get(f"{server_url}/broadcast")
    number = resp.json()['number']
    return number

def upload(client_id,number):
    resp = requests.post(f"{server_url}/collect",json={"client_id":client_id,"number":number})
    if resp.status_code != 200:
        print("Error uploading data")
    elif "ready" in resp.json():
        print("Data uploaded successfully, aggregation complete")
    elif "wait" in resp.json():
        client_left = resp.json()['wait']
        print(f"Data uploaded successfully, waiting for {client_left} more client(s)")
if __name__ == "__main__":
    print("Client 1 started")
    client_id = 1
    own_number = 1
    received_number = 0
    choice =0
    while True:
        print()
        print("Enter 0 for get, 1 for train, 2 for upload")
        action = enter(choice)
        print(f"Action: {action}")
        time.sleep(3)
        
        if action == "get":
            received_number = get_broadcast()
            print(f"Received number: {received_number}")
        if action == "train":
            new_number = own_number + received_number
            print(f"New number: {new_number}")
        if action == "upload":
            upload(client_id,new_number)
        if action == "fuck":
            print("go back to fucking work")
        choice +=1