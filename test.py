import geocoder
g = geocoder.ip('me')
print(g.latlng)

dict = {'Haroon': 5}
for key, count in dict.items():
    print(key, count)

text = 'abc || s'
print(text.split('||'))