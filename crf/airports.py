import os

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_airport_names():
    airport_names_list = [
        set(), set(), set(), set(), set(), set(), set(), set(), set(), set(), set()
    ]
    with open(os.path.join(_CURRENT_DIR, 'airports.csv')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            airport_name = line.split(',')[3].strip('"')

            airport_name_tokens = airport_name.split(' ')

            for j, token in enumerate(airport_name_tokens):
                airport_names_list[j].add(token.lower())

    return airport_names_list


def get_airport_codes():
    airport_codes_set = set()
    with open(os.path.join(_CURRENT_DIR, 'airport_codes.txt')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            airport_codes_set.add(line.split('\t')[0])

    return airport_codes_set
