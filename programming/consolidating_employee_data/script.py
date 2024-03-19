
"""
You just got hired as the first and only data practitioner at a small business experiencing exponential growth.
The company needs more structured processes, guidelines, and standards.
Your first mission is to structure the human resources data.
The data is currently scattered across teams and files and comes in various formats: Excel files, CSVs, JSON files...

Following is the data in the `datasets` folder:
- __Office addresses__
    - Saved in office_addresses.csv.
    - If the value for office is `NaN`, then the employee is remote.
- __Employee addresses__
    - Saved on the first tab of `employee_information.xlsx`.
- __Employee emergency contacts__
    - Saved on the second tab of `employee_information.xlsx`; this tab is called `emergency_contacts`.
    - However, this sheet was edited at some point, and ***the headers were removed***!
    The HR manager let you know that they should be:
        `employee_id`, `last_name`, `first_name`, `emergency_contact`, `emergency_contact_number`, and `relationship`.
- __Employee roles, teams, and salaries__
    - This information has been exported from the company's human resources management system into a JSON file titled
    `employee_roles.json`. Here are the first few lines of that file:

    {"A2R5H9":
      {
        "title": "CEO",
        "monthly_salary": "$4500",
        "team": "Leadership"
      },
     ...
    }
    
"""




import pandas as pd

# df contains office addresses
addresses_office = pd.read_csv('./datasets/office_addresses.csv')

# df contains employee addresses
addresses_emp = pd.read_excel('./datasets/employee_information.xlsx')

emergency_contacts_header = ["employee_id", "last_name", "first_name",
                             "emergency_contact", "emergency_contact_number", "relationship"]

final_columns = ["employee_id", "employee_first_name", "employee_last_name", "employee_country",
                 "employee_city", "employee_street", "employee_street_number", "emergency_contact",
                 "emergency_contact_number", "relationship", "monthly_salary", "team", "title", "office",
                 "office_country", "office_city", "office_street", "office_street_number"]

# df contains employee emergency contacts details
emergency_contacts = pd.read_excel('./datasets/employee_information.xlsx',
                                    sheet_name='emergency_contacts',
                                    header=None,
                                    names=emergency_contacts_header)

# df contains employee, team, roles details
teams = pd.read_json('./datasets/employee_roles.json', orient='index')

# merging df based on unique val but with different colnames
employees_temp = addresses_emp.merge(addresses_office, left_on='employee_country', right_on='office_country',how='left')

employees_temp = employees_temp.merge(emergency_contacts, on='employee_id')

employees_final = employees_temp.merge(teams, left_on='employee_id', right_on=teams.index, how='left')

# df with columns to keep
employees_final = employees_final[final_columns]

# replacing NA vals with "Remote" in selected cols
for col in ["office", "office_country", "office_city", "office_street", "office_street_number"]:
    employees_final[col].fillna('Remote', inplace=True)

employees_final = employees_final.set_index('employee_id')



