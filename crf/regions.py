import os

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_region_names():
    region_names_list = [
        set(), set(), set(), set(), set(), set()
    ]
    with open(os.path.join(_CURRENT_DIR, 'regions.csv')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            region_name = line.split(',')[3].strip('"')

            region_name_tokens = region_name.split(' ')

            for j, token in enumerate(region_name_tokens):
                region_names_list[j].add(token.lower())

    return region_names_list


def get_region_codes():
    region_codes_set = set()
    with open(os.path.join(_CURRENT_DIR, 'regions.csv')) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            region_code = line.split(',')[2].strip('"')
            try:
                int(region_code)
            except ValueError:
                # code must not be a number
                region_codes_set.add(region_code)

    return region_codes_set
