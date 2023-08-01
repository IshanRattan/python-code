

from cryptography.fernet import Fernet
import sys

def generate_key():
    return Fernet.generate_key()

def encrypt_file(file_path, key):
    fernet = Fernet(key)
    with open(file_path, 'rb') as file:
        file_data = file.read()
    encrypted_data = fernet.encrypt(file_data)
    with open(file_path + '.enc', 'wb') as file:
        file.write(encrypted_data)
    return file_path + '.enc'

def main(file_path):

    key = generate_key()

    encrypted_file = encrypt_file(file_path, key)
    print(f"Encrypted file: {encrypted_file}")
    print(f"Encryption key: {key.decode()}")

if __name__ == "__main__":
    main('encrypt.py')
