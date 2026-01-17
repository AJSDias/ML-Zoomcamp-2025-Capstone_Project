import requests

url = "http://127.0.0.1:8000/predict"

def test_predict(path_to_image):
    
    files = {"file": open(path_to_image, "rb")}

    resp = requests.post(url, files=files)
    print(resp.json())

    return resp.json()

test_predict("./data/test/PNEUMONIA/person1_virus_6.jpeg")