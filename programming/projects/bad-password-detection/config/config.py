
### Params ###

# Regex
upperCase = '[A-Z]'
lowerCase = '[a-z]'
alphaNumeric = '\W'
numeric = '\d'
firstName = '(^\w+)'
lastName = '(\w+$)'
checkString = 'password'

# Data Import
csvPath = 'data/users.csv'

# Initialize obj
userNames = []
results = {'total': 0,
           'valid': 0,
           'invalid': 0}

