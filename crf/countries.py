import os

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_country_names():
    country_names_list = [
        set(), set(), set(), set(), set(), set(), set()
    ]
    with open(os.path.join(_CURRENT_DIR, 'countries.csv')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            country_name = line.split(',')[2].strip('"')
            country_name_tokens = country_name.split(' ')

            for j, token in enumerate(country_name_tokens):
                country_names_list[j].add(token.lower())

    return country_names_list


def get_country_codes():
    country_codes_set = set()
    with open(os.path.join(_CURRENT_DIR, 'countries.csv')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            country_codes_set.add(line.split(',')[1].strip('"'))

    return country_codes_set
