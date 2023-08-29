import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from tqdm import tqdm

#input_file = "spec_spw25_27_25_gauss_prop.tsv"
database_file = "tmp/combined_database_CDMS_patched.txt"
#prop_df = pd.read_csv(input_file, sep='\t')
# prop_df['mean'].plot.hist(bins=50)
# prop_df['amplitude'].plot.hist(bins=50)
# prop_df['fwhm'].plot.hist(bins=50)
# plt.show()


def parse_string(s):
    d = {
        "id": int(s[0 : 6]),
        "name": s[7 : 33],
        "freq": float(s[35 : 48]),
        "uncertain": float(s[48 : 56]),
        "integ_intens": float(s[56 : 64]),
        "DoF": int(s[64 : 66]),
        "LSE": float(s[66 : 76]),
        "Gup": s[76 : 79],
        "tag": int(s[79 : 86]),
        "q_code": int(s[86 : 90]),
        "q": s[90 : ]
    }

    return d

def parse_string_re(input_string):
    pattern = r"(\d{6})\s+(.*)\s*\|\s*(\d*\.\d{4}|\d*\.\d{3})\s*(\d*\.\d{4}|\d*\.\d{3})\s*(-?\d*\.\d{4})\s*(\d*)\s+"

    match = re.match(pattern, input_string)

    if match:
        data = match.groups()
        number1 = data[0]
        string1 = data[1]
        number2 = float(data[2])
        number3 = float(data[3])
        number4 = float(data[4])
        number5 = int(data[5])


        return number1, string1, number2, number3, number4, number5



    print(f'cannot parse the string: {input_string}')
    return None

database_list = []

with open(database_file, "r") as file:
    for line in tqdm(file):
        d = parse_string(line.strip())
        database_list.append(d)



database_df = pd.DataFrame(database_list)
database_df.to_csv('./database_csv_patched.csv', index_label='trans_id')

