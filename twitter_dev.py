import getpass
import os
import bcrypt
import twitter

from aes_cipher import AESCipher
from constants import *


class TwitterDev:
    """Twitter developer object that automatically loads related information"""

    def __init__(self, data_path):
        while True:
            try:
                with open(data_path, 'rb') as f:
                    hashed_password = f.read(BCRYPT_LENGTH)
                    encrypted_consumer_key = f.read(48)
                    encrypted_consumer_secret = f.read(80)
                    encrypted_access_token_key = f.read(80)
                    encrypted_access_token_secret = f.read(80)

                break
            except OSError as e:
                print('Data of Twitter developer is not accessible: ' + str(e))
                if not self.prompt_init():
                    raise Exception('Cannot create Twitter developer data at this time.')

        auth_success = False
        max_try = 3
        password = ''
        for auth_try in (1, max_try):
            password = getpass.getpass('Password: ', stream = None)
            if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                print('Password is not correct. (' + str(auth_try) + '/' + str(max_try) + ')')
                continue
            auth_success = True
            break

        if not auth_success:
            raise Exception('Authentication failed!')
        print('Authentication granted!')

        cipher = AESCipher(password)
        consumer_key = cipher.decrypt(encrypted_consumer_key)
        consumer_secret = cipher.decrypt(encrypted_consumer_secret)
        access_token_key = cipher.decrypt(encrypted_access_token_key)
        access_token_secret = cipher.decrypt(encrypted_access_token_secret)
        self.api = twitter.Api(consumer_key, consumer_secret, access_token_key, access_token_secret)
        credentials = self.api.VerifyCredentials()
        print(credentials)

    @staticmethod
    def prompt_init():
        while True:
            answer = input('Do you want to (re)initialize a new Twitter developer account? (Y/n) ')
            if answer == 'n' or answer == 'N':
                return False
            elif answer == 'y' or answer == 'Y' or len(answer) == 0:
                TwitterDev.__data_fill()
                return True
            else:
                print('Answer is not recognizable. Please try again.')

    @staticmethod
    def __data_fill():
        username = input('Username: ')
        password = getpass.getpass('Password: ', stream = None)
        consumer_key = input('Consumer Key: ')
        consumer_secret = input('Consumer Secret: ')
        access_token_key = input('Access Token Key: ')
        access_token_secret = input('Access Token Secret: ')
        cipher = AESCipher(password)
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        data_path = DATA_FOLDER + '/' + username
        with open(data_path, 'wb') as data:
            data.write(bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()))
            data.write(cipher.encrypt(consumer_key))
            data.write(cipher.encrypt(consumer_secret))
            data.write(cipher.encrypt(access_token_key))
            data.write(cipher.encrypt(access_token_secret))
