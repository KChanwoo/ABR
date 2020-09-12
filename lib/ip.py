import json
import requests

class IPLocation:
    @staticmethod
    def get_region(ip):
        if ip == "127.0.0.1":
            return "local"

        URL = 'https://ipinfo.io/{}/json'.format(ip)
        response = requests.get(URL)
        location = json.loads(response.text)

        return "{}, {}, {}".format(location["org"], location["city"], location["country"])
