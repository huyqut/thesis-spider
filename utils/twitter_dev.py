import os
import getpass
import bcrypt
import twitter
from utils.aes_cipher import AESCipher


class TwitterDev:
    """Twitter developer object that automatically loads related information"""

    def __init__(self, data_path):
        while True:
            try:
                with open(data_path, 'r') as f:
                    lines = f.readlines()
                    hashed_password = lines[0]
                    encrypted_consumer_key = lines[1]
                    encrypted_consumer_secret = lines[2]
                    encrypted_access_token_key = lines[3]
                    encrypted_access_token_secret = lines[4]
                break
            except OSError as e:
                print('Data of Twitter developer is not accessible: ' + str(e))
                if not self.__prompt_init():
                    raise Exception('Cannot create Twitter developer data at this time.')

        auth_success = False
        max_try = 3
        password = ''
        for auth_try in (1, max_try):
            password = getpass.getpass('Password: ', stream = None)
            if not bcrypt.checkpw(password, hashed_password):
                print('Password is not correct. (' + str(auth_try) + '/' + str(max_try) + ')')
                continue
            auth_success = True

        if not auth_success:
            raise Exception('Authentication failed!')
        print('Authentication granted!')

        cipher = AESCipher(password)
        consumer_key = cipher.decrypt(encrypted_consumer_key)
        consumer_secret = cipher.decrypt(encrypted_consumer_secret)
        access_token_key = cipher.decrypt(encrypted_access_token_key)
        access_token_secret = cipher.decrypt(encrypted_access_token_secret)
        self.api = twitter.Api(consumer_key, consumer_secret, access_token_key, access_token_secret)

    def __prompt_init(self):
        while True:
            answer = input('Do you want to (re)initialize a new Twitter developer account? (Y/n) ')
            if answer == 'n' or answer == 'N':
                return False
            elif answer == 'y' or answer == 'Y' or len(answer) == 0:
                self.__data_fill()
                return True
            else:
                print('Answer is not recognizable. Please try again.')

    def __data_fill(self):
        username = input('Username: ')
        password = getpass.getpass('Password: ', stream = None)
        consumer_key = input('Consumer Key: ')
        consumer_secret = input('Consumer Secret: ')
        access_token_key = input('Access Token Key: ')
        access_token_secret = input('Access Token Secret: ')
        cipher = AESCipher(password)
        data_folder = '.data'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        data_path = data_folder + '/' + username
        with open(data_path, 'w') as data:
            data.write(bcrypt.hashpw(password, bcrypt.gensalt()) + '\n')
            data.write(cipher.encrypt(consumer_key) + '\n')
            data.write(cipher.encrypt(consumer_secret) + '\n')
            data.write(cipher.encrypt(access_token_key) + '\n')
            data.write(cipher.encrypt(access_token_secret) + '\n')
