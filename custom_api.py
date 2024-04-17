import requests

class DeepCall:
    def do(query: str = ""):
        headers = {
            "Accept": "application/json",
            "x-token": "token"
        }

        url = "http://demoapi.yipzale.me/logout"

        response = requests.post(url=url, headers=headers).json()
        if response['code'] == 0:
            return "成功"
        else:
            return response['message']

if __name__ == "__main__":
    message = DeepCall.do("")
    print(message)