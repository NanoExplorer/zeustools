def get_value(hkfile, key):
    with open(hkfile, 'r') as the_file:
        data = the_file.readlines()
    for i, d in enumerate(data):
        if key in d:
            arr = d.split(":")
            return arr[1].strip()


def modify_value(hkfile, key, value):
    with open(hkfile, 'r') as the_file:
        data = the_file.readlines()
    for i, d in enumerate(data):
        if key in d:
            arr = d.split(":")
            arr[1] = value
            data[i] = f"{arr[0]}: {arr[1]}\n"

    with open(hkfile, 'w') as the_file:
        for line in data:
            the_file.write(line)
