import contextlib
import logging
from functools import wraps
from time import sleep

from bulb_credentials import IP, PASSWORD, USERNAME

try:
    from PyP100 import PyL530
except ImportError:
    PyL530 = None

RED = 0
GREEN = 120
REPEATS = 5
WAIT = 2

def exception_exit_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info('Exception exit handler')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info(f'handling exception {e}')
            handle_exception()
        finally:
            logging.info('finishing up')
            handle_exit()
    return wrapper

def handle_exception():
    with contextlib.suppress(Exception):
        bulb = PyL530.L530(IP, PASSWORD, USERNAME)
        bulb.handshake()
        bulb.login()

        for _ in range(REPEATS):
            bulb.setColor(RED, 100)
            sleep(WAIT)
            bulb.setColorTemp(2700)
            sleep(WAIT)

def handle_exit():
    with contextlib.suppress(Exception):
        bulb = PyL530.L530(IP, PASSWORD, USERNAME)
        bulb.handshake()
        bulb.login()

        for _ in range(REPEATS):
            bulb.setColor(GREEN, 100)
            sleep(WAIT)
            bulb.setColorTemp(2700)
            sleep(WAIT)
