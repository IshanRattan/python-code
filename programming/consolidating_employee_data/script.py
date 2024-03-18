
import pandas as pd

addresses_office = pd.read_csv('./datasets/office_addresses.csv')
addresses_emp = pd.read_excel('./datasets/employee_information.xlsx')

emergency_contacts_header = ["employee_id", "last_name", "first_name",
                             "emergency_contact", "emergency_contact_number", "relationship"]

final_columns = ["employee_id", "employee_first_name", "employee_last_name", "employee_country",
                 "employee_city", "employee_street", "employee_street_number", "emergency_contact",
                 "emergency_contact_number", "relationship", "monthly_salary", "team", "title", "office",
                 "office_country", "office_city", "office_street", "office_street_number"]

emergency_contacts = pd.read_excel('./datasets/employee_information.xlsx',
                                    sheet_name='emergency_contacts',
                                    header=None,
                                    names=emergency_contacts_header)

teams = pd.read_json('./datasets/employee_roles.json', orient='index')

employees_temp = addresses_emp.merge(addresses_office, left_on='employee_country', right_on='office_country',how='left')

employees_temp = employees_temp.merge(emergency_contacts, on='employee_id')

employees_final = employees_temp.merge(teams, left_on='employee_id', right_on=teams.index, how='left')

employees_final = employees_final[final_columns]

for col in ["office", "office_country", "office_city", "office_street", "office_street_number"]:
    employees_final[col].fillna('Remote', inplace=True)

employees_final = employees_final.set_index('employee_id')



