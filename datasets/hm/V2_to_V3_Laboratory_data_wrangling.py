import datetime as dt
import os
from pathlib import Path

import pandas as pd

######################################### WRANGLING LABORATORY TABLE DATA #########################################
V2_data_folder = Path("./raw_data/")

V2_laboratory = V2_data_folder / "04_实验室检查.csv"

### V2 CDSL_04 - Data wrangling: removing lines exceeding fields number (7) and change delimiter from ";"  to ","
data = []

# try: # open file in read mode
fp = open("datasets/hm/raw_data/04_实验室检查.csv", "r", newline="\r\n", encoding="latin1")

for line in fp:
    line = line.replace(";", ",")
    data.append(line)

# finally:
fp.close()


# try: # open file in writing mode
f = open("datasets/hm/raw_data/hm_labtest.csv", "w", newline="\n", encoding="utf-8")
for line in data:
    commas_num = line.count(",")

    if (
        commas_num <= 7
    ):  # Removing lines with more than 7 fields # Condition for data to be deleted
        f.write(line)
# finally:
f.close()
