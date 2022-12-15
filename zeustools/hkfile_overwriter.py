import argparse


if __name__=='__main__':

    p = argparse.ArgumentParser(
            prog="hkfile_overwriter",
            description="Overwrite certain parameters in hk files (e.g. in case grating index is missing)"
        )
    p.add_argument('key')
    p.add_argument('new_value')
    p.add_argument('filename', nargs="*")

    args = p.parse_args()

    for f in args.filename:
        with open(f, 'r') as the_file:
            data = the_file.readlines()
        for i,d in enumerate(data):
            if args.key in d:
                arr = d.split(":")
                arr[1] = args.new_value
                data[i] = f"{arr[0]}: {arr[1]}\n"
                break
        with open(f, 'w') as the_file:
            for line in data:
                the_file.write(line)

