from functools import wraps
from time import sleep

from PyP100 import PyL530

from src.bulb_credentials import IP, PASSWORD, USERNAME

RED = 0
GREEN = 120
REPEATS = 2

def exception_exit_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('Exception exit handler')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'handling exception {e}')
            handle_exception()
        finally:
            print('finishing up')
            handle_exit()
    return wrapper

def handle_exception():
    bulb = PyL530.L530(IP, PASSWORD, USERNAME)
    bulb.handshake()
    bulb.login()

    for _ in range(REPEATS):
        bulb.setColor(RED, 100)
        sleep(2)
        bulb.setColorTemp(2700)
        sleep(2)

def handle_exit():
    bulb = PyL530.L530(IP, PASSWORD, USERNAME)
    bulb.handshake()
    bulb.login()

    for _ in range(REPEATS):
        bulb.setColor(GREEN, 100)
        sleep(2)
        bulb.setColorTemp(2700)
        sleep(2)
