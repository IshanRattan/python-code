
# Parameters to validate:
# Password must have at least 10 characters
# Must have(at least):
    # one lower case character
    # one upper case character
    # one numeric character
    # one non-alphanumeric character
    # must not contain the "password" string
    # must not contain first or last name of the user.

# Following are the columns in the dataset used:
    # id: userwise unique ID.
    # username: username with the format firstname.lastname.
    # password: password set by users.


# Package import.
import pandas as pd
import re

def hasPattern(pattern : str, string : str) -> bool:
    """Requires regex string pattern & input string to find if the pattern exists in the input string"""
    return True if re.search(pattern, string) else False

def validatePwd(username: str, password: str) -> bool:
    """Requires username & password strings to validate
    if the password is strong or weak as per NIST guidelines"""

    # Calc length of password. IF length < 10, password invalid, no further validations.
    charLen = len(password)
    if charLen >= 10:
        # Check if password has atleast one upper case char.
        hasUpperCase = hasPattern('[A-Z]', password)

        # Check if password has atleast one lower case char.
        hasLowerCase = hasPattern('[a-z]', password)

        # Check if password has atleast one numeric.
        hasNumeric = hasPattern('\d', password)

        # Check if password has atleast one alphanumeric.
        hasAlphaNum = hasPattern('\W', password)

        # Check if password has "password" string in it.
        noPwdStr = True if 'password' not in password.lower() else False

        # Fetch firstname & lastname from username.
        firstname, lastname = re.findall('(^\w+)', username)[0], re.findall('(\w+$)', username)[0]

        # Check if firstname or lastname is present in user password.
        nameInPwd = True if firstname.lower() in password.lower() or lastname.lower() in password.lower() else False
        
        # Condition to check if above rules are met or not.
        if hasUpperCase and hasLowerCase and hasNumeric and hasAlphaNum and noPwdStr and not nameInPwd:
            return True
        else:
            return False
    else:
        return False


# Initialize "usernames" empty list & "results" dictionary.
usernames = []
results = {'total': 0,
           'valid': 0,
           'invalid': 0}

# Load user data for usernames & passwords.
loginData = pd.read_csv('data/users.csv')
print(loginData.head(4))

# Iterate over the loaded dataset & validate the password.
# Update "usernames" & "results" in the each iteration.
for idx, row in loginData.iterrows():
    if validatePwd(row['username'], row['password']):
        results['total'] += 1
        results['valid'] += 1
    else:
        results['total'] += 1
        results['invalid'] += 1
        usernames.append(row['username'])

# Create a pandas series of Usernames with bad passwords
userList = pd.Series(user for user in usernames)
userList = userList.sort_values()

# Calc % of passwords that do not meet NIST standards. (weakPasswords / totalPasswords) * 100
weakPasswordPercent = round(results['invalid'] / results['total'], 2)
print('Percentage(%) of users with weak password:', weakPasswordPercent * 100)

