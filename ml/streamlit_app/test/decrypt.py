
from cryptography.fernet import Fernet
import sys

def decrypt_file(encrypted_file_path, decrypted_file_path, key):
    with open(encrypted_file_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()

    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)

    with open(decrypted_file_path, 'wb') as decrypted_file:
        decrypted_file.write(decrypted_data)

def decrypt(encrypted_file_path,
            decrypted_file_path,
            key):

    decrypt_file(encrypted_file_path, decrypted_file_path, key)

decrypt('/Users/ishanrattan/Desktop/Study/github/python-code/ml/test/encrypt.py.enc',
        '/Users/ishanrattan/Desktop/Study/github/python-code/ml/test/encrypt-new.py',
        'WjxNss9-aiDpQtbmeXKHDOoFysgdnD56VAlqEpPSpvU=')