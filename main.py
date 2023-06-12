import sys


from UI.main import run_ui

CONFIG_FILE = 'user.config'
BASE_URL = "http://34.142.141.69/" #'http://127.0.0.1:8000/'
#BASE_URL = "http://34.66.216.13"
RTMP_URL = "rtmp://34.142.141.69/live/"

if __name__ == "__main__":

    app, config = run_ui(CONFIG_FILE, BASE_URL, RTMP_URL)
    app.exec_()
    print('HIIIII')
    sys.exit()


