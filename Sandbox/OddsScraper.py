import cloudscraper
import base64

scraper = cloudscraper.create_scraper()

url = 'https://www.bestfightodds.com/api/ggd?m=12250&p=1'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'Referer': 'https://www.bestfightodds.com/',
    'X-Requested-With': 'XMLHttpRequest'
}

response = scraper.get(url, headers=headers)
data = response.content.decode().replace('-', '+').replace('_', '/')

def custom_decode(data):
    data = base64.b64decode(data)
    data = data.decode('latin1')

    g = r"""!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    decoded = ''
    for char in data:
        idx = g.find(char)
        if idx >= 0:
            char = g[(idx + len(g)//2) % len(g)]
        decoded += char
    return decoded


decoded_string = custom_decode(data)
print(decoded_string)

