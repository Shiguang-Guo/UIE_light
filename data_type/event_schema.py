"""
@author: Guo Shiguang
@software: PyCharm
@file: event_schema.py
@time: 2022/3/9 13:04
"""
import json


class EventSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        type_list = [item.lower() for item in type_list]
        role_list = [item.lower() for item in role_list]
        type_role_dict = [item.lower() for item in type_role_dict]
        return EventSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')
            output.write(json.dumps(self.type_role_dict) + '\n')
