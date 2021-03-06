import base64
import hashlib
from Crypto.Cipher import AES
from Crypto import Random


class AESCipher(object):
    """Reference: http://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256"""

    def __init__(self, key):
        self.bs = 32
        self.key = hashlib.sha256(key.encode('utf-8')).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        #enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        raw = cipher.decrypt(enc[AES.block_size:])
        return self._unpad(raw).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
